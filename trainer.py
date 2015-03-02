import cPickle
from collections import defaultdict
import glob
import os
import re
import numpy
import scipy
from sklearn.linear_model import LogisticRegression, SGDClassifier, LassoLarsCV
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import Normalizer, PolynomialFeatures
from data import DataHandler
from league import League, PointSpreads

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
        self.daynum = daynum
        self.team_one = team_one
        self.team_two = team_two

    def features(self):
        if self.daynum is None:
            features = []
            for team in (self.team_one, self.team_two):
                features += max(v for k, v in team.features.iteritems() if k[0] == self.season)
            return features

        key = (self.season, self.daynum)
        try:
            return self.team_one.features[key] + self.team_two.features[key]
        except KeyError:
            return None


class AllFeatures:
    def __init__(self):
        self.label_pickle = os.path.join(PICKLE_DIR, '{:d}_labels.pkl')
        self.feature_pickle = os.path.join(PICKLE_DIR, '{:d}_features.pkl')
        self._db = DataHandler()
        self.league = League()
        self.pointspreads = PointSpreads()

    def build_features(self):
        for season in range(2007, 2015):
            self.features_and_labels(season)

    def features_and_labels(self, season):
        feature_pickle = self.feature_pickle.format(season)
        label_pickle = self.label_pickle.format(season)
        if os.path.exists(feature_pickle) and os.path.exists(label_pickle):
            return cPickle.load(open(feature_pickle)), cPickle.load(open(label_pickle))

        with self._db.connector() as cur:
            cur.execute("""SELECT daynum, wteam, lteam
                    FROM  regular_season_compact_results
                    WHERE season = ?""", (season,))

            features = []
            labels = []
            print(season)
            for j, row in enumerate(cur):
                print(j)
                team_ids = row['wteam'], row['lteam']
                line = [self.pointspreads.pred_game(season, row['wteam'], row['lteam'], row['daynum'])]
                team_objs = [self.league.data(team_id) for team_id in team_ids]

                # Model should be symmetric, so add features for team A vs team B and team B vs team A
                team_features = Features(season, team_objs[0], team_objs[1], row['daynum']).features()
                if team_features:  # Features returns false-y value if there isn't enough data
                    features.append(team_features + line)
                    labels.append(1)

                    team_features = Features(season, team_objs[1], team_objs[0], row['daynum']).features()
                    features.append(team_features + [-j for j in line])
                    labels.append(0)

            cPickle.dump(features, open(feature_pickle, 'w'))
            cPickle.dump(labels, open(label_pickle, 'w'))
        return features, labels

    @staticmethod
    def clean():
        map(os.remove, glob.glob(os.path.join(PICKLE_DIR, "*")))


def log_loss(y, y_hat):
    epsilon = 1e-15
    y = numpy.array(y)
    y_hat = scipy.minimum(1 - epsilon, scipy.maximum(epsilon, numpy.array(y_hat)))
    return -(y * scipy.log(y_hat) + (1 - y) * scipy.log(1 - y_hat)).mean()


def features_labels(before_season):
    features, labels = [], []
    all_features = AllFeatures()
    for season in range(2007, before_season):
        season_features, season_labels = all_features.features_and_labels(season)
        features += season_features
        labels += season_labels
    return numpy.array(features), numpy.array(labels)


def find_best_model(season):
    raw_train_x, train_y = features_labels(season)
    raw_test_x, test_y = map(numpy.array, AllFeatures().features_and_labels(season))
    normalizer = Normalizer()
    train_x = normalizer.fit_transform(raw_train_x)
    test_x = normalizer.transform(raw_test_x)

    results = {}
    max_right = (0, 1)
    least_loss = 1
    use_pca = True

    # for use_pca in (True, False):
    # if use_pca:
    # pca = PCA(n_components=12)
    # train_x = pca.fit_transform(raw_train_x)
    # test_x = pca.transform(raw_test_x)
    # else:
    #         train_x = raw_train_x
    #         test_x = raw_test_x
    #
    # for loss_func in ("log", "modified_huber"):
    #     for penalty in ("l1", "l2"):
    #         for alpha in [10 ** (0.5 * j) for j in range(-8, 0)]:
    #             model = SGDClassifier(loss=loss_func, penalty=penalty, alpha=alpha, n_iter=1000).fit(train_x,
    #                                                                                                  train_y)
    #             key = ("sgd", use_pca, loss_func, penalty, 2 * numpy.log10(alpha))
    #             max_right, least_loss = evaluate(model, key, test_x, test_y, results, max_right, least_loss)

    # test logistic regressions
    for penalty in ("l1", "l2"):
        for c in [10 ** (0.5 * j) for j in range(4, 16)]:
            model = LogisticRegression(penalty=penalty, C=c).fit(train_x, train_y)
            key = (penalty, use_pca, 2 * numpy.log10(c))
            max_right, least_loss = evaluate(model, key, test_x, test_y, results, max_right, least_loss)

    model = BaggingClassifier(LogisticRegression(penalty=penalty, C=c), n_estimators=100,
                              max_samples=0.5).fit(train_x, train_y)
    key = (penalty + " bagging", use_pca, 2 * numpy.log10(c))
    max_right, least_loss = evaluate(model, key, test_x, test_y, results, max_right, least_loss)

    print("{:d} out of {:d}".format(*map(int, max_right)))
    return results


def evaluate(model, key, test_x, test_y, results, max_right, least_loss):
    predictions = model.predict_proba(test_x)[:, 1]

    num_right = (test_y == predictions.round()).sum()
    pct_right = num_right / float(len(test_y))
    if num_right >= max_right[0]:
        max_right = (num_right, len(test_y))

    results[key] = log_loss(test_y, predictions)
    if results[key] < least_loss:
        least_loss = results[key]
    print("{:s}\n\t{:.1f}% right ({:,d} out of {:,d})\n\t{:.5f}".format(
        str(key), 100 * pct_right, num_right, len(test_y), results[key]))
    return max_right, least_loss


def main():
    results = defaultdict(list)
    test_years = range(2010, 2015)
    for year in test_years:
        print(year)
        year_results = find_best_model(year)
        for model, loss in year_results.items():
            results[model].append(loss)
        model, score = min(year_results.items(), key=lambda j: j[1])
        print("Model: {:s}\n\tScore: {:.5f}\n".format(str(model), score))

    for j, model in enumerate(sorted(results.items(), key=lambda j: numpy.mean(j[1]))):
        print("{:,d}. {:s}".format(j + 1, print_model(model)))

    best_models = [min(results.items(), key=lambda j: numpy.median(j[1])),
                   min(results.items(), key=lambda j: numpy.mean(j[1])),
                   min(results.items(), key=lambda j: numpy.min(j[1])),
                   min(results.items(), key=lambda j: numpy.max(j[1])),
    ]

    print("\nBest models:")
    for model in best_models:
        print(print_model(model))


def print_model(model):
    return "\nName: {:s}\n\tMean score:\t{:.5f}\n\tMedian score:\t{:.5f}\n\tMin score:\t{:.5f}\n\tMax score:\t{:.5f}".format(
        str(model[0]),
        numpy.mean(model[1]),
        numpy.median(model[1]),
        numpy.min(model[1]),
        numpy.max(model[1]),
    )


def reset():
    AllFeatures.clean()
    AllFeatures().build_features()


def scratch():
    season = 2005
    raw_train_x, train_y = features_labels(season)
    raw_test_x, test_y = map(numpy.array, AllFeatures().features_and_labels(season))
    normalizer = Normalizer()
    poly = PolynomialFeatures(degree=2, interaction_only=True)
    train_x = normalizer.fit_transform(poly.fit_transform(raw_train_x))
    test_x = normalizer.transform(poly.transform(raw_test_x))
    penalty = 'l2'
    c = 10 ** 4
    model = LogisticRegression(penalty=penalty, C=c).fit(train_x, train_y)
    key = (penalty, 2 * numpy.log10(c))
    predictions = model.predict_proba(test_x)
    print(model.coef_)
    print(predictions)
    max_right = evaluate(model, key, test_x, test_y, {}, (0, 1))


if __name__ == '__main__':
    main()
