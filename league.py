import numpy
from scipy.linalg import norm
from data import DataHandler
from team import Team


def extract_metric(row, team_id, metric):
    return row['w' + metric] if row['wteam'] == team_id else row['l' + metric]


class League:
    def __init__(self, season):
        self.season = season
        self._db = DataHandler()
        self._team_idxs = None
        self._team_ids = None
        self._pagerank = None
        self._team_data = {}

    def data(self, team_id):
        if team_id not in self._team_data:
            self._team_data[team_id] = Team(team_id, self.season)
        return self._team_data[team_id]

    def team_metric_delta(self, team_id, metric):
        diffs = []
        for game in self.data(team_id).data:
            prefix = 'l' if team_id == game['wteam'] else 'w'
            other_team = self.data(game[prefix + 'team'])
            this_game = extract_metric(game, other_team.id, metric)
            other_games = [extract_metric(row, other_team.id, metric) for row in other_team.data]
            other_games.remove(this_game)
            diffs.append(this_game - numpy.median(other_games))
        return numpy.median(diffs)

    def team_deltas(self, team_id):
        metrics = (
            "fga",
            "fgm",
            "fga3",
            "fgm3",
            "fta",
            "ftm",
            "pf",
            "to",
            "ast",
            "stl",
            "or",
            "dr",
            "blk",
            "score",
        )
        return [self.team_metric_delta(team_id, metric) for metric in metrics]

    def _lookups(self):
        self._team_idxs = {}
        self._team_ids = {}
        with self._db.connector() as cur:
            cur.execute("""SELECT wteam, lteam
                               FROM regular_season_compact_results
                               WHERE season = ?""", (self.season,))
            for row in cur:
                if row["wteam"] not in self._team_idxs:
                    idx = len(self._team_idxs)
                    self._team_idxs[row["wteam"]] = idx
                    self._team_ids[idx] = row["wteam"]
                if row["lteam"] not in self._team_idxs:
                    idx = len(self._team_idxs)
                    self._team_idxs[row["lteam"]] = idx
                    self._team_ids[idx] = row["lteam"]

    def strength(self, team_id):
        return self.pagerank[self.team_idxs[team_id]]


    @property
    def team_ids(self):
        if self._team_ids is None:
            self._lookups()
        return self._team_ids

    @property
    def team_idxs(self):
        if self._team_idxs is None:
            self._lookups()
        return self._team_idxs

    @property
    def pagerank(self):
        if self._pagerank is None:
            idxs = self.team_idxs
            A = numpy.zeros((len(idxs), len(idxs)))
            with self._db.connector() as cur:
                cur.execute("""SELECT wteam, lteam, wscore - lscore AS spread
                               FROM regular_season_compact_results
                               WHERE season = ?""", (self.season,))
                for row in cur:
                    # A[idxs[row['wteam']], idxs[row['lteam']]] += row['spread']
                    A[idxs[row['wteam']], idxs[row['lteam']]] = 1
            # normalize
            col_sums = A.sum(1)
            col_sums[col_sums == 0] = 1
            A /= col_sums
            new_x = numpy.zeros((A.shape[0],))
            new_y = numpy.ones((A.shape[0],))
            while norm(new_x - new_y) > 0.0001:
                new_x = new_y
                new_y = numpy.dot(A, new_y)
                new_y /= norm(new_y)
            self._pagerank = new_y
        return self._pagerank


def main():
    season = League(2009)
    print(season.team_metric_delta(1314, 'fga3'))


if __name__ == '__main__':
    main()
