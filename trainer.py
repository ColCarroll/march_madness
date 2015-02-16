import cPickle
from collections import defaultdict
import os
import re
import numpy
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from data import DataHandler
from team import Team
from league import League

DIR = os.path.dirname(os.path.realpath(__file__))
PICKLE_DIR = os.path.join(DIR, "pickles")

if not os.path.exists(PICKLE_DIR):
    os.mkdir(PICKLE_DIR)


def int_seed(str_seed):
    match = re.search("(\d+)", str_seed)
    if match:
        return int(match.group(1))
    else:
        print(str_seed)
        return 16


class Features:
    def __init__(self, season, team_one, team_two, daynum=None):
        self.season = season
        self.team_one = team_one
        self.team_two = team_two
        self.daynum = daynum

    def features(self):
        return self.team_one.features() + self.team_two.features()


class SeasonFeatures:
    def __init__(self, season):
        self.label_pickle = os.path.join(PICKLE_DIR, '{:d}_labels.pkl'.format(season))
        self.feature_pickle = os.path.join(PICKLE_DIR, '{:d}_features.pkl'.format(season))
        self.season = season
        self._db = DataHandler()
        self.league = League(season)

    def features_and_labels(self):
        if os.path.exists(self.feature_pickle) and os.path.exists(self.label_pickle):
            return cPickle.load(open(self.feature_pickle)), cPickle.load(open(self.label_pickle))

        with self._db.connector() as cur:
            cur.execute("""SELECT tcr.season, ts1.seed AS wseed, tcr.wteam, ts2.seed AS lseed, tcr.lteam
                    FROM  tourney_compact_results tcr
                    JOIN tourney_seeds ts1 ON tcr.wteam = ts1.team AND tcr.season = ts1.season
                    JOIN tourney_seeds ts2 ON tcr.lteam = ts2.team AND tcr.season = ts2.season
                    WHERE tcr.season = ?""", (self.season,))

            features = []
            labels = []
            teams = {}
            for j, row in enumerate(cur):
                print(j)

                team_ids = row['wteam'], row['lteam']
                seeds = list(map(int_seed, [row['wseed'], row['lseed']]))
                pageranks = list(map(self.league.strength, team_ids))

                team_objs = []
                for team_id in team_ids:
                    key = (self.season, team_id)
                    if key not in teams:
                        teams[key] = Team(team_id, self.season)
                    team_objs.append(teams[key])

                # Model should be symmetric, so add features for team A vs team B and team B vs team A
                team_features = Features(row['season'], team_objs[0], team_objs[1]).features()
                features.append(team_features + seeds + pageranks)
                labels.append(1)

                team_features = Features(row['season'], team_objs[1], team_objs[0]).features()
                features.append(team_features + seeds[-1::-1] + pageranks[-1::-1])
                labels.append(0)

                cPickle.dump(features, open(self.feature_pickle, 'w'))
                cPickle.dump(labels, open(self.label_pickle, 'w'))
        return features, labels

    def clean(self):
        os.remove(self.feature_pickle)
        os.remove(self.label_pickle)


def log_loss(y, y_hat):
    epsilon = 1e-15
    y = numpy.array(y)
    y_hat = scipy.minimum(1 - epsilon, scipy.maximum(epsilon, numpy.array(y_hat)))
    return -(y * scipy.log(y_hat) + (1 - y) * scipy.log(1 - y_hat)).mean()


def features_labels(before_season):
    features, labels = [], []
    for season in range(2003, before_season):
        season_features, season_labels = SeasonFeatures(season).features_and_labels()
        features += season_features
        labels += season_labels
    return numpy.array(features), numpy.array(labels)


def find_best_model(season):
    train_x, train_y = features_labels(season)
    test_x, test_y = map(numpy.array, SeasonFeatures(season).features_and_labels())
    results = {}
    #
    # # test random forests
    # depth = 3
    # for learning_rate in [10 ** (0.5 * j) for j in range(-8, -1)]:
    # model = GradientBoostingClassifier(n_estimators=500, learning_rate=learning_rate, max_depth=depth).fit(
    # train_x, train_y)
    #     predictions = model.predict_proba(test_x)[:, 1]
    #     results[("random forest", depth, learning_rate)] = log_loss(test_y, predictions)

    # test logistic regressions
    max_right = (0, 1)
    for penalty in ("l1", "l2"):
        for c in [10 ** (0.5 * j) for j in range(-15, 0)]:
            model = LogisticRegression(penalty=penalty, C=c).fit(train_x, train_y)
            predictions = model.predict_proba(test_x)[:, 1]

            num_right = (test_y == predictions.round()).sum()
            pct_right = num_right / float(len(test_y))
            if pct_right > float(max_right[0]) / max_right[1]:
                max_right = (num_right, len(test_y))

            results[(penalty, c)] = log_loss(test_y, predictions)
    print("{:d} out of {:d}".format(*map(int, max_right)))
    return results


def main():
    results = defaultdict(list)
    test_years = range(2005, 2015)
    for year in test_years:
        print(year)
        year_results = find_best_model(year)
        for model, loss in year_results.items():
            results[model].append(loss)
        print("Score: {:.5f}\n".format(min(year_results.items(), key=lambda j: j[1])[1]))

    best_models = [min(results.items(), key=lambda j: numpy.median(j[1])),
                   min(results.items(), key=lambda j: numpy.mean(j[1])),
                   min(results.items(), key=lambda j: numpy.min(j[1])),
                   min(results.items(), key=lambda j: numpy.max(j[1])),
    ]

    print("\nBest models:")
    for model in best_models:
        print("\nName: {:s}\nMean score:\t{:.5f}\nMedian score:\t{:.5f}\nMin score:\t{:.5f}\nMax score:\t{:.5f}".format(
            str(model[0]),
            numpy.mean(model[1]),
            numpy.median(model[1]),
            numpy.min(model[1]),
            numpy.max(model[1]),
        )
        )


if __name__ == '__main__':
    main()