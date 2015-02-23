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


class MedianFeature:
    def __init__(self, label, data, metric, team_id):
        self.data = data
        self.label = label
        self.id = team_id
        self.win_metric = "w" + metric
        self.lose_metric = "l" + metric
        self._for = None
        self._against = None

    def _execute(self):
        self._for = numpy.median(
            [j[self.win_metric] if j['wteam'] == self.id else j[self.lose_metric] for j in self.data]
        )
        self._against = numpy.median(
            [j[self.lose_metric] if j['wteam'] == self.id else j[self.win_metric] for j in self.data]
        )

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

    def __init__(self, team_id, season):
        self.id = team_id
        self.season = season
        self._data = None
        self._db = DataHandler()
        self.fgm = MedianFeature("Field goals made", self.data, 'fgm', self.id)
        self.fga = MedianFeature("Field goals attempted", self.data, 'fga', self.id)
        self.fgm3 = MedianFeature("3 point field goals made", self.data, 'fgm3', self.id)
        self.fga3 = MedianFeature("3 point field goals attempted", self.data, 'fga3', self.id)
        self.ftm = MedianFeature("Free throws made", self.data, 'ftm', self.id)
        self.fta = MedianFeature("Free throws attempted", self.data, 'fta', self.id)

        self.all_features = [QueryFeature("Name", self._db, "SELECT team_name FROM teams WHERE team_id=?", self.id),
                             QueryFeature("Wins", self._db, """
                SELECT
                    COUNT(*)
                FROM
                    regular_season_compact_results
                WHERE
                    season = ?
                AND
                    wteam = ?""", self.season, self.id),
                             QueryFeature("Losses", self._db, """
                SELECT
                    COUNT(*)
                FROM
                    regular_season_compact_results
                WHERE
                    season = ?
                AND
                    lteam = ?""", self.season, self.id),
                             MedianFeature("Score", self.data, 'score', self.id),
                             self.fgm,
                             self.fga,
                             self.fgm3,
                             self.fga3,
                             self.ftm,
                             self.fta,
                             MedianFeature("Offensive rebounds", self.data, 'or', self.id),
                             MedianFeature("Defensive rebounds", self.data, 'dr', self.id),
                             MedianFeature("Assists", self.data, 'ast', self.id),
                             MedianFeature("Turnovers", self.data, 'to', self.id),
                             MedianFeature("Steals", self.data, 'stl', self.id),
                             MedianFeature("Blocks", self.data, 'blk', self.id),
                             MedianFeature("Personal fouls", self.data, 'pf', self.id)] + \
                            self._ratios() + self._polls()

    @property
    def data(self):
        if self._data is None:
            with self._db.connector() as cur:
                cur.execute("""
                    SELECT
                        *
                    FROM
                        regular_season_detailed_results
                    WHERE
                        season = ?
                    AND
                        (wteam = ? OR lteam = ?)""", (self.season, self.id, self.id)
                )
                self._data = [row for row in cur]
        return self._data

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
                if data:
                    first_rank = min(data, key=lambda j: j['rating_day_num'])['orank']
                    last_rank = max(data, key=lambda j: j['rating_day_num'])['orank']
                    rank_ratio = float(first_rank) / float(last_rank)
                else:
                    first_rank, last_rank, rank_ratio = 0, 0, 0
                polls.extend([
                    BaseFeature("{:s} poll start".format(poll), first_rank),
                    BaseFeature("{:s} poll end".format(poll), last_rank),
                    BaseFeature("{:s} poll ratio".format(poll), rank_ratio)
                ])
        return polls

    def features(self):
        features = []
        for ftr in self.all_features[1:]:
            if isinstance(ftr, MedianFeature):
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
