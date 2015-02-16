import numpy
from collections import namedtuple
from data import DataHandler

POLLS = [  # these appear to be the most consistent
           "MOR",
           "POM",
           "SAG",
]


class BaseFeature:
    def __init__(self, label, val):
        self.label = label
        self.val = val

    def __repr__(self):
        return str(self.label)

    def __str__(self):
        return "{:s}:\t{:s}".format(*map(str, [self.label, self.val]))


class QueryFeature:
    def __init__(self, label, db, query, *args):
        self._db = db
        self._val = None
        self.label = label
        self.query = query
        self.args = args

    @property
    def val(self):
        if self._val is None:
            with self._db.connector(commit=False) as cur:
                cur.execute(self.query, self.args)
                try:
                    self._val = next(cur)[0]
                except StopIteration:
                    raise ValueError("No results returned!")
        return self._val

    def __repr__(self):
        return str(self.label)

    def __str__(self):
        return "{:s}:\t{:s}".format(str(self.label), str(self.val))


class QueryMedian:
    def __init__(self, label, db, metric, team_id, season, day=None):
        self._db = db
        self.label = label
        self.metric = metric
        self.id = team_id
        self.season = season
        self.day = day
        self._for = None
        self._against = None

    def query(self):
        return """
        SELECT
            CASE wteam WHEN ? THEN w{0:s} ELSE l{0:s} END AS for,
            CASE lteam WHEN ? THEN w{0:s} ELSE l{0:s} END AS against
        FROM
            regular_season_detailed_results
        WHERE
            season = ?
        AND
            (wteam = ? OR lteam = ?)
        AND
            daynum < COALESCE(?, 1000)""".format(self.metric)

    def _execute(self):
        _for = []
        _against = []
        with self._db.connector(commit=False) as cur:
            cur.execute(self.query(), (self.id, self.id, self.season, self.id, self.id, self.day))
            for row in cur:
                _for.append(row["for"])
                _against.append(row["against"])
        self._for = numpy.median(_for)
        self._against = numpy.median(_against)

    @property
    def for_(self):
        if self._for is None:
            self._execute()
        return self._for

    @property
    def against(self):
        if self._against is None:
            self._execute()
        return self._against

    def __repr__(self):
        return str(self.label)

    def __str__(self):
        return "{:s} for:\t{:s}\n{:s} against:\t{:s}".format(
            *map(str, [self.label, self.for_, self.label, self.against]))


class Team:
    pseudo_feature = namedtuple("pseudo_feature", ["for_", "against"])

    def __init__(self, team_id, season, day=None):
        self.id = team_id
        self.season = season
        self.day = day
        self._db = DataHandler()
        self.fgm = QueryMedian("Field goals made", self._db, 'fgm', self.id, self.season)
        self.fga = QueryMedian("Field goals attempted", self._db, 'fga', self.id, self.season)
        self.fgm3 = QueryMedian("3 point field goals made", self._db, 'fgm3', self.id, self.season)
        self.fga3 = QueryMedian("3 point field goals attempted", self._db, 'fga3', self.id, self.season)
        self.ftm = QueryMedian("Free throws made", self._db, 'ftm', self.id, self.season)
        self.fta = QueryMedian("Free throws attempted", self._db, 'fta', self.id, self.season)

        self.all_features = [QueryFeature("Name", self._db, "SELECT team_name FROM teams WHERE team_id=?", self.id),
                             QueryFeature("Wins", self._db, """
                SELECT
                    COUNT(*)
                FROM
                    regular_season_compact_results
                WHERE
                    season = ?
                AND
                    wteam = ?
                AND
                    daynum < COALESCE(?, 1000)""", self.season, self.id, self.day),
                             QueryFeature("Losses", self._db, """
                SELECT
                    COUNT(*)
                FROM
                    regular_season_compact_results
                WHERE
                    season = ?
                AND
                    lteam = ?
                AND
                    daynum < COALESCE(?, 1000)""", self.season, self.id, self.day),
                             QueryMedian("Score", self._db, 'score', self.id, self.season),
                             self.fgm,
                             self.fga,
                             self.fgm3,
                             self.fga3,
                             self.ftm,
                             self.fta,
                             QueryMedian("Offensive rebounds", self._db, 'or', self.id, self.season),
                             QueryMedian("Defensive rebounds", self._db, 'dr', self.id, self.season),
                             QueryMedian("Assists", self._db, 'ast', self.id, self.season),
                             QueryMedian("Turnovers", self._db, 'to', self.id, self.season),
                             QueryMedian("Steals", self._db, 'stl', self.id, self.season),
                             QueryMedian("Blocks", self._db, 'blk', self.id, self.season),
                             QueryMedian("Personal fouls", self._db, 'pf', self.id, self.season)] + \
                            self._ratios() + self._polls()

    def __hash__(self):
        return hash((self.season, self.id, self.day))

    def _ratios(self):
        return [
            BaseFeature("Field goal pct for", self.fgm.for_ / float(self.fga.for_)),
            BaseFeature("Field goal pct against", self.fgm.against / float(self.fga.against)),
            BaseFeature("3 pt field goal pct for", self.fgm3.for_ / float(self.fga3.for_)),
            BaseFeature("3 pt field goal pct against", self.fgm3.against / float(self.fga3.against)),
            BaseFeature("Free throw pct for", self.ftm.for_ / float(self.fta.for_)),
            BaseFeature("Free throw pct against", self.ftm.against / float(self.fta.against)),
        ]

    def _polls(self):
        polls = []
        for poll in POLLS:
            with self._db.connector() as cur:
                cur.execute("""SELECT rating_day_num, orank
                               FROM massey_ordinals
                               WHERE season=? AND team=? AND sys_name=?;""", (self.season, self.id, poll))
                data = list(cur)
                first_rank = min(data, key=lambda j: j['rating_day_num'])['orank']
                last_rank = max(data, key=lambda j: j['rating_day_num'])['orank']
                rank_ratio = float(first_rank) / float(last_rank)
                polls.extend([
                    BaseFeature("{:s} poll start".format(poll), first_rank),
                    BaseFeature("{:s} poll end".format(poll), last_rank),
                    BaseFeature("{:s} poll ratio".format(poll), rank_ratio)
                ])
        return polls

    def features(self):
        features = []
        for ftr in self.all_features[1:]:
            if isinstance(ftr, QueryMedian):
                features.append(ftr.for_)
                features.append(ftr.against)
            elif isinstance(ftr, QueryFeature) or isinstance(ftr, BaseFeature):
                features.append(ftr.val)
        return features

    def __repr__(self):
        return "{:d} {:d}".format(self.season, self.id)

    def __str__(self):
        return "\n".join(str(feature) for feature in self.all_features)


if __name__ == '__main__':
    team = Team(1314, 2005)
    print(str(team))
