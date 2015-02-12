import numpy
from collections import namedtuple
from data import DataHandler


class QueryFeature:
    def __init__(self, db, query, *args):
        self._db = db
        self._val = None
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
        return self.val

    def __str__(self):
        return str(self.val)


class QueryMedian:
    def __init__(self, db, metric, team_id, season, day=None):
        self._db = db
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


class Team:
    pseudo_feature = namedtuple("pseudo_feature", ["for_", "against"])

    def __init__(self, team_id, season, day=None):
        self.id = team_id
        self.season = season
        self.day = day
        self._db = DataHandler()
        self.name = QueryFeature(self._db, "SELECT team_name FROM teams WHERE team_id=?", self.id)
        # features
        self.wins = QueryFeature(self._db, """
        SELECT
            COUNT(*)
        FROM
            regular_season_compact_results
        WHERE
            season = ?
        AND
            wteam = ?
        AND
            daynum < COALESCE(?, 1000)""", self.season, self.id, self.day)
        self.losses = QueryFeature(self._db, """
        SELECT
            COUNT(*)
        FROM
            regular_season_compact_results
        WHERE
            season = ?
        AND
            lteam = ?
        AND
            daynum < COALESCE(?, 1000)""", self.season, self.id, self.day)

        self.field_goals_made = QueryMedian(self._db, 'fgm', self.id, self.season)
        self.field_goals_attempted = QueryMedian(self._db, 'fga', self.id, self.season)
        self.three_pt_field_goals_made = QueryMedian(self._db, 'fgm3', self.id, self.season)
        self.three_pt_field_goals_attempted = QueryMedian(self._db, 'fga3', self.id, self.season)
        self.free_throws_made = QueryMedian(self._db, 'ftm', self.id, self.season)
        self.free_throws_attempted = QueryMedian(self._db, 'fta', self.id, self.season)
        self.offensive_rebounds = QueryMedian(self._db, 'or', self.id, self.season)
        self.defensive_rebounds = QueryMedian(self._db, 'dr', self.id, self.season)
        self.assists = QueryMedian(self._db, 'ast', self.id, self.season)
        self.turnovers = QueryMedian(self._db, 'to', self.id, self.season)
        self.steals = QueryMedian(self._db, 'stl', self.id, self.season)
        self.blocks = QueryMedian(self._db, 'blk', self.id, self.season)
        self.personal_fouls = QueryMedian(self._db, 'pf', self.id, self.season)
        self._points_for = None
        self._points_against = None

    def __repr__(self):
        return "{:d} {:s}".format(self.season, self.name)

    @property
    def field_goal_pct(self):
        return self.pseudo_feature(
            for_=float(self.field_goals_made.for_) / float(self.field_goals_attempted.for_),
            against=float(self.field_goals_made.against) / float(self.field_goals_attempted.against))

    def _points(self):
        self._points_for = []
        self._points_against = []
        with self._db.connector(commit=False) as cur:
            cur.execute("""
        SELECT
            CASE wteam WHEN ? THEN wscore ELSE lscore END AS points_for,
            CASE lteam WHEN ? THEN wscore ELSE lscore END AS points_against
        FROM
            regular_season_compact_results
        WHERE
            season = ?
        AND
            (wteam = ? OR lteam = ?)""", (self.id, self.id, self.season, self.id, self.id))
            for row in cur:
                self._points_for.append(row["points_for"])
                self._points_against.append(row["points_against"])

    @property
    def avg_points_for(self):
        if self._points_for is None:
            self._points()
        return numpy.median(self._points_for)

    @property
    def avg_points_against(self):
        if self._points_against is None:
            self._points()
        return numpy.median(self._points_against)

    def __str__(self):
        return "{:d} {:s}\n{:d}-{:d}\nAvg Points For: {:.1f}\tAgainst: {:.1f}\n" \
               "Average field goals for: {:.1f}\tAgainst: {:.1f}\n" \
               "Average field goals attempted: {:.1f}\tAttempted against: {:.1f}\n" \
               "Average field goal pct: {:.2f}\tPct against: {:.2f}\n" \
               "Average free throws for: {:.1f}\tAgainst: {:.1f}\n" \
               "Average free throws attempted: {:.1f}\tAttempted against: {:.1f}\n" \
               "Average three pointers for: {:.1f}\tAgainst: {:.1f}\n" \
               "Average three pointers attempted: {:.1f}\tAttempted against: {:.1f}\n" \
               "Average offensive rebounds for: {:.1f}\tAgainst: {:.1f}\n" \
               "Average defensive rebounds for: {:.1f}\tAgainst: {:.1f}\n" \
               "Average assists for: {:.1f}\tAgainst: {:.1f}\n" \
               "Average turnovers for: {:.1f}\tAgainst: {:.1f}\n" \
               "Average steals for: {:.1f}\tAgainst: {:.1f}\n" \
               "Average blocks for: {:.1f}\tAgainst: {:.1f}\n" \
               "Average personal fouls for: {:.1f}\tAgainst: {:.1f}".format(
            self.season, self.name,
            self.wins.val, self.losses.val,
            self.avg_points_for, self.avg_points_against,
            self.field_goals_made.for_, self.field_goals_made.against,
            self.field_goals_attempted.for_, self.field_goals_attempted.against,
            self.field_goal_pct.for_, self.field_goal_pct.against,
            self.free_throws_made.for_, self.free_throws_made.against,
            self.free_throws_attempted.for_, self.free_throws_attempted.against,
            self.three_pt_field_goals_made.for_, self.three_pt_field_goals_made.against,
            self.three_pt_field_goals_attempted.for_, self.three_pt_field_goals_attempted.against,
            self.offensive_rebounds.for_, self.offensive_rebounds.against,
            self.defensive_rebounds.for_, self.defensive_rebounds.against,
            self.assists.for_, self.assists.against,
            self.turnovers.for_, self.turnovers.against,
            self.steals.for_, self.steals.against,
            self.blocks.for_, self.blocks.against,
            self.personal_fouls.for_, self.personal_fouls.against,
        )


if __name__ == '__main__':
    for season in range(2003, 2015):
        print(str(Team(1314, season, 20)))
        print("\n")
        # team = Team(1314, 2003)
        # print(str(team))
