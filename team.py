"""
The polls that have the most data, in order
"SAG"	"12"	"6114.0"
"MOR"	"12"	"6031.5"
"POM"	"12"	"5823.91666666667"
"WLK"	"12"	"4862.16666666667"
"BOB"	"12"	"4700.25"
"DOL"	"12"	"4594.16666666667"
"COL"	"12"	"4252.08333333333"
"RPI"	"12"	"4039.0"
"WOL"	"12"	"3966.58333333333"
"RTH"	"12"	"3922.0"
"USA"	"12"	"454.666666666667"
"AP"	"12"	"454.416666666667"
"""

POLLS = ('SAG', 'MOR', 'POM', 'WLK', 'BOB', 'DOL', 'COL', 'RPI', 'WOL', 'RTH', 'USA', 'AP')

STATS = (
    'fga',
    'fgm',
    'fta',
    'ftm',
    'fga3',
    'fgm3',
    'stl',
    'ast',
    'dr',
    'or',
    'blk',
    'pf',
    'to'
)

import numpy
from data import DataHandler


class AggregatorCollector:
    def __init__(self, aggregators):
        self.aggregators = {}
        for aggregator in aggregators:
            self.aggregators[aggregator.label] = aggregator

    def __getitem__(self, item):
        return self.aggregators[item]

    def update(self, game, team_obj):
        prefix = 'w' if game['wteam'] == team_obj.id else 'l'
        for aggregator in self.aggregators.itervalues():
            aggregator.update(game, team_obj, prefix)


class Aggregator:
    def __init__(self, label, func):
        self.label = label
        self._func = func
        self.season = None
        self.val = []

    def reset(self):
        self.val = []

    def update(self, game, team_obj, prefix):
        if game['season'] != self.season:
            self.season = game['season']
            self.reset()
        self.val.append(self._func(game, team_obj, prefix, self.val))

    @property
    def value(self):
        return numpy.median(self.val[:-1])


def wins(game, team, prefix, val):
    return val + int(prefix == 'w')


def losses(game, team, prefix, val):
    return val + int(prefix == 'l')


def stat_agg(stat):
    def agg(game, team, prefix, val):
        return game["{:s}{:s}".format(prefix, stat)]

    return agg


def pct_agg(num_stat, denom_stat):
    def agg(game, team, prefix, val):
        return float(game["{:s}{:s}".format(prefix, num_stat)]) / max(1.0, float(game["{:s}{:s}".format(prefix, denom_stat)]))
    return agg


class Team:
    def __init__(self, team_id):
        self.id = team_id
        self._db = DataHandler()
        self._data = None
        self._ranks = None
        self._name = None
        self._features = None
        self.aggregator = AggregatorCollector([Aggregator(stat, stat_agg(stat)) for stat in STATS] +
                                              [Aggregator('fgpct', pct_agg('fga', 'fgm')),
                                               Aggregator('fgpct3', pct_agg('fga3', 'fgm3')),
                                               Aggregator('ftpct', pct_agg('fta', 'ftm'))])

    @property
    def ranks(self):
        if self._ranks is None:
            with self._db.connector() as cur:
                cur.execute("""
                    SELECT
                        orank, season, rating_day_num, sys_name
                    FROM
                        massey_ordinals
                    WHERE
                        team = ?
                    AND
                        sys_name IN ({:s})
                    ORDER BY
                        season, rating_day_num""".format(",".join("'{:s}'".format(poll) for poll in POLLS)), (self.id,))
                self._ranks = list(cur)
        return self._ranks

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
                        (wteam = ? OR lteam = ?)
                    ORDER BY
                        season, daynum""", (self.id, self.id))
                self._data = list(cur)
        return self._data

    def is_after_first_n_games(self, game, n):
        return sum(1 for j in self.data if j['season'] == game['season'] and j['daynum'] < game['daynum']) > n

    def get_rank_during_game(self, game):
        ranks = {j: 0 for j in POLLS}
        for row in self.ranks:
            if row['season'] == game['season']:
                if row['rating_day_num'] < game['daynum']:
                    ranks[row['sys_name']] = row['orank']
        ranks = numpy.array(ranks.values())
        ranks = ranks[ranks > 0]
        if len(ranks) == 0:
            return numpy.log(351)  # highest possible rank
        return numpy.log(numpy.median(ranks))

    def _get_wins(self, game):
        return sum(int(row['wteam'] == self.id) for row in self.data if
                   row['season'] == game['season'] and row['daynum'] < game['daynum'])

    @property
    def name(self):
        if self._name is None:
            with self._db.connector() as cur:
                cur.execute("""SELECT team_name FROM teams WHERE team_id = ?""", (self.id,))
                self._name = list(cur)[0][0]
        return self._name

    @property
    def features(self):
        if self._features is None:
            self._features = {}
            for game in self.data:
                self.aggregator.update(game, self)
                if self.is_after_first_n_games(game, 5):
                    aggs = self.aggregator.aggregators
                    key = (game['season'], game['daynum'])
                    self._features[key] = [
                                              self.get_rank_during_game(game),
                                              self._get_wins(game),
                                          ] + [agg.value for agg in aggs.values()] + [

                                          ]
        return self._features

    def __repr__(self):
        return "Team {:d}".format(self.id)

    def __str__(self):
        return self.name


if __name__ == '__main__':
    team = Team(1314)
    print(str(team))
    for k, v in sorted(team.features.items()):
        print(k, v)
    for key, value in team.aggregator.aggregators.iteritems():
        print(key, value.value)
