import numpy
from scipy.linalg import norm
from data import DataHandler
from team import Team


class League:
    def __init__(self, season, day=None):
        self.season = season
        self.day = day
        self._db = DataHandler()
        self._team_idxs = None
        self._team_ids = None
        self._pagerank = None

    def _lookups(self):
        self._team_idxs = {}
        self._team_ids = {}
        with self._db.connector() as cur:
            cur.execute("""SELECT wteam, lteam
                               FROM regular_season_compact_results
                               WHERE season = ? AND daynum < COALESCE(?, 1000)""", (self.season, self.day))
            for row in cur:
                if row["wteam"] not in self._team_idxs:
                    idx = len(self._team_idxs)
                    self._team_idxs[row["wteam"]] = idx
                    self._team_ids[idx] = row["wteam"]
                if row["lteam"] not in self._team_idxs:
                    idx = len(self._team_idxs)
                    self._team_idxs[row["lteam"]] = idx
                    self._team_ids[idx] = row["lteam"]

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
                               WHERE season = ? AND daynum < COALESCE(?, 1000)""", (self.season, self.day))
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
    season = League(2014)
    ranks = numpy.argsort(season.pagerank)[-1::-1]
    for j, team_id in enumerate(ranks[:10]):
        print("{:,d}. {:s}\n".format(j + 1, str(Team(season.team_ids[team_id], season.season))))


if __name__ == '__main__':
    main()
