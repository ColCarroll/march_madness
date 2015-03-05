import numpy
from scipy.linalg import norm
from sklearn.linear_model import LassoLarsCV
from data import DataHandler
from team import Team

FIRST_SEASON = 2007

def extract_metric(row, team_id, metric):
    return row['w' + metric] if row['wteam'] == team_id else row['l' + metric]


class PointSpreads:
    lines = (
        "linesage",
        "linedok",
        "linepugh",
        "linesag",
        "linemoore",
        "linesagp",
        "linefox",
        "linepom",
        "linepig"
    )

    def __init__(self):
        self._data = None
        self._db = DataHandler()
        self._seasons = {}

    @property
    def data(self):
        if self._data is None:
            with self._db.connector() as cur:
                cur.execute("""SELECT * from pointspreads;""")
                self._data = list(cur)
        return self._data

    def pred_game(self, season, team_one, team_two, daynum=None):
        if season < FIRST_SEASON:
            return 0
        model = self.pred_season(season)
        for row in [j for j in self.data if j["season"] == season]:
            if row['daynum'] == daynum or daynum is None:
                if row['wteam'] == team_one and row['lteam'] == team_two:
                    return model.predict(self.get_feature(row))
                if row['wteam'] == team_two and row['lteam'] == team_one:
                    return model.predict([-j for j in self.get_feature(row)])
        return 0

    def get_feature(self, row):
        feature = []
        for line in [row[line] for line in self.lines]:
            try:
                feature.append(float(line))
            except ValueError:
                feature.append(0)
        return feature

    def pred_season(self, season):
        if season not in self._seasons:
            features = []
            labels = []
            for row in [j for j in self.data if j["season"] < season]:
                feature = self.get_feature(row)

                features.append(feature)
                labels.append(row["wscore"] - row["lscore"])

                features.append([-j for j in feature])
                labels.append(-row["wscore"] + row["lscore"])
            self._seasons[season] = LassoLarsCV(fit_intercept=False).fit(features, labels)
        return self._seasons[season]


class League:
    def __init__(self):
        self._db = DataHandler()
        self._team_idxs = {}
        self._team_ids = {}
        self._pagerank = {}
        self._team_data = {}
        self._pointspreads = {}

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

    def strength(self, season, team_id):
        return self.pagerank(season)[self.team_idxs(season)[team_id]]

    def team_ids(self, season):
        if season not in self._team_ids:
            self._lookups(season)
        return self._team_ids[season]

    def team_idxs(self, season):
        if season not in self._team_idxs:
            self._lookups(season)
        return self._team_idxs[season]

    def pagerank(self, season):
        if season not in self._pagerank:
            idxs = self.team_idxs(season)
            A = numpy.zeros((len(idxs), len(idxs)))
            with self._db.connector() as cur:
                cur.execute("""SELECT wteam, lteam, wscore - lscore AS spread
                               FROM regular_season_compact_results
                               WHERE season = ?""", (season,))
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
            self._pagerank[season] = new_y
        return self._pagerank[season]


def main():
    spreads = PointSpreads()
    for season in range(FIRST_SEASON, 2015):
        print(season)
        spreads.pred_season(season)


if __name__ == '__main__':
    main()
