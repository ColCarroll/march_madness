import csv
import os
import datetime
from data import DataHandler, NameMap

from sklearn.linear_model import LassoLarsCV

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR, "DATA")
FIRST_SEASON = 2007


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
                self._data = list(cur) + CurrentPointspreads().data()
        return self._data

    def pred_game(self, season, team_one, team_two, daynum=None):
        if season < FIRST_SEASON:
            return 0
        model = self.pred_season(season)
        season_data = [j for j in self.data if j["season"] == season]
        if daynum is None:
            games = [j for j in season_data if {j['wteam'], j['lteam']} == {team_one, team_two}]
            if games:
                most_recent = max(games, key=lambda j: int(j["daynum"]))
                if most_recent['wteam'] == team_one:
                    return model.predict(self.get_feature(most_recent))
                return model.predict([-j for j in self.get_feature(most_recent)])

        else:
            for row in [j for j in season_data if j["daynum"] == daynum]:
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
                feature.append(0.0)
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


class CurrentPointspreads:
    fname = os.path.join(DATA_DIR, "ncaabb14.csv")
    dayzero = datetime.date(2015, 11, 3)
    date_fmt = "%m/%d/%Y"

    def __init__(self):
        self._names = NameMap()

    def data(self):
        data = []
        with open(self.fname) as buff:
            reader = csv.DictReader(buff)
            for row in reader:
                home = self._names.lookup(row["home"])
                road = self._names.lookup(row["road"])
                if home and road:
                    row["wteam"] = home
                    row["lteam"] = road
                    row['daynum'] = (datetime.datetime.strptime(row["date"], self.date_fmt).date() - self.dayzero).days
                    row["season"] = 2015
                    data.append(row)
        return data


def main():
    p = PointSpreads().pred_game(2015, 1246, 1438)
    print(p)


if __name__ == '__main__':
    main()