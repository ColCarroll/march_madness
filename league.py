import numpy
from scipy.linalg import norm
from data import DataHandler
from team import Team
from pointspreads import PointSpreads

FIRST_SEASON = 2007


def extract_metric(row, team_id, metric):
    return row['w' + metric] if row['wteam'] == team_id else row['l' + metric]


class League:
    def __init__(self):
        self._db = DataHandler()
        self._pointspreads = PointSpreads()
        self._team_idxs = {}
        self._team_ids = {}
        self._pagerank = {}
        self._team_data = {}

    def data(self, team_id):
        if team_id not in self._team_data:
            self._team_data[team_id] = Team(team_id)
        return self._team_data[team_id]

    def _lookups(self, season):
        self._team_idxs[season] = {}
        self._team_ids[season] = {}
        with self._db.connector() as cur:
            cur.execute("""SELECT wteam, lteam FROM regular_season_compact_results where season = ?""", (season,))
            for row in cur:
                if row["wteam"] not in self._team_idxs[season]:
                    idx = len(self._team_idxs[season])
                    self._team_idxs[season][row["wteam"]] = idx
                    self._team_ids[season][idx] = row["wteam"]
                if row["lteam"] not in self._team_idxs[season]:
                    idx = len(self._team_idxs[season])
                    self._team_idxs[season][row["lteam"]] = idx
                    self._team_ids[season][idx] = row["lteam"]

    def team_ids(self, season):
        if season not in self._team_ids:
            self._lookups(season)
        return self._team_ids[season]

    def team_idxs(self, season):
        if season not in self._team_idxs:
            self._lookups(season)
        return self._team_idxs[season]

    def strength(self, team_id, season, daynum=None):
        return self.pagerank(season, daynum)[self.team_idxs(season)[team_id]]

    def pointspread(self, season, team_one, team_two, daynum=None):
        return self._pointspreads.pred_game(season, team_one, team_two, daynum)

    def pagerank(self, season, daynum=None):
        if daynum is None:
            daynum = 1000
        if daynum not in self._pagerank.get(season, {}):
            idxs = self.team_idxs(season)
            A = numpy.zeros((len(idxs) + 1, len(idxs) + 1))
            A[-1, :] = 1
            A[:, -1] = 1
            with self._db.connector() as cur:
                cur.execute("""SELECT wteam, lteam, wscore - lscore AS spread
                               FROM regular_season_compact_results
                               WHERE season = ? and daynum < ?""", (season, daynum))
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
            if season not in self._pagerank:
                self._pagerank[season] = {}
            self._pagerank[season][daynum] = new_y
        return self._pagerank[season][daynum]


def main():
    pass


if __name__ == '__main__':
    main()
