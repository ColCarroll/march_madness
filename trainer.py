import cPickle
from collections import defaultdict
import csv
import glob
import os
import re
import numpy
import scipy
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import Normalizer
from data import DataHandler
from league import League, PointSpreads

FIRST_SEASON = 2003
DIR = os.path.dirname(os.path.realpath(__file__))
PICKLE_DIR = os.path.join(DIR, "pickles")
OUT_DIR = os.path.join(DIR, 'out_data')

if not os.path.exists(PICKLE_DIR):
    os.mkdir(PICKLE_DIR)

if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)


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


class TourneyFeatures:
    pred_dir = os.path.join(OUT_DIR, 'predictions')

    def __init__(self, season):
        self._db = DataHandler()
        self.season = season
        self.league = League()
        self.pointspreads = PointSpreads()
        self.pred_path = os.path.join(self.pred_dir, '{:d}.csv'.format(season))

    def tourney_teams(self):
        with self._db.connector() as cur:
            cur.execute("SELECT team FROM tourney_seeds WHERE season = ?", (self.season,))
            team_ids = sorted([j[0] for j in cur])
        return team_ids

    def get_features_and_ids(self):
        features = []
        ids = []
        team_ids = self.tourney_teams()
        for j, team_one_id in enumerate(team_ids):
            for team_two_id in team_ids[j:]:
                team_one = self.league.data(team_one_id)
                team_two = self.league.data(team_two_id)
                features.append(team_features(team_one, team_two, self.season, self.pointspreads))
                ids.append("{:d}_{:d}_{:d}".format(self.season, team_one_id, team_two_id))
        return numpy.array(features), ids

    def write_predictions(self, model, transform):
        if not os.path.exists(self.pred_dir):
            os.mkdir(self.pred_dir)

        features, ids = self.get_features_and_ids()
        predictions = model.predict_proba(transform(features))[:, 1]
        with open(self.pred_path, 'w') as buff:
            buff.write("id,pred\n")
            for (label, pred) in zip(ids, predictions):
                buff.write("{:s},{:s}\n".format(label, str(pred)))

    def score_predictions(self):
        if not os.path.exists(self.pred_path):
            return 0

        pred_dict = {}
        with open(self.pred_path, 'r') as buff:
            reader = csv.DictReader(buff)
            for row in reader:
                pred_dict[row['id']] = float(row['pred'])

        predictions = []
        labels = []
        with self._db.connector() as cur:
            cur.execute("SELECT season, wteam, lteam from tourney_compact_results where season=?", (self.season,))
            for row in cur:
                if row[1] < row[2]:
                    labels.append(1)
                    predictions.append(pred_dict["{:d}_{:d}_{:d}".format(self.season, row['wteam'], row['lteam'])])
                else:
                    labels.append(0)
                    predictions.append(pred_dict["{:d}_{:d}_{:d}".format(self.season, row['lteam'], row['wteam'])])
        return log_loss(labels, predictions)


class AllFeatures:
    def __init__(self):
        self.label_pickle = os.path.join(PICKLE_DIR, '{:d}_labels.pkl')
        self.feature_pickle = os.path.join(PICKLE_DIR, '{:d}_features.pkl')
        self._db = DataHandler()
        self.league = League()
        self.pointspreads = PointSpreads()

    def build_features(self):
        for season in range(FIRST_SEASON, 2015):
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
                wteam = self.league.data(row['wteam'])
                lteam = self.league.data(row['lteam'])
                game_features = team_features(wteam, lteam, season, self.pointspreads, row['daynum'])
                if game_features:
                    features.append(game_features)
                    labels.append(1)
                    features.append(team_features(lteam, wteam, season, self.pointspreads, row['daynum']))
                    labels.append(0)
            cPickle.dump(features, open(feature_pickle, 'w'))
            cPickle.dump(labels, open(label_pickle, 'w'))
        return features, labels

    @staticmethod
    def clean():
        map(os.remove, glob.glob(os.path.join(PICKLE_DIR, "*")))


def team_features(team_one, team_two, season, pointspread_obj, daynum=None):
    line = [pointspread_obj.pred_game(season, team_one.id, team_two.id, daynum)]
    game_features = Features(season, team_one, team_two, daynum).features()
    if game_features:
        return game_features + line


def log_loss(y, y_hat):
    epsilon = 1e-15
    y = numpy.array(y)
    y_hat = scipy.minimum(1 - epsilon, scipy.maximum(epsilon, numpy.array(y_hat)))
    return -(y * scipy.log(y_hat) + (1 - y) * scipy.log(1 - y_hat)).mean()


def features_labels(before_season):
    features, labels = [], []
    all_features = AllFeatures()
    for season in range(FIRST_SEASON, before_season):
        season_features, season_labels = all_features.features_and_labels(season)
        features += season_features
        labels += season_labels
    return numpy.array(features), numpy.array(labels)


def find_best_model(season, fname):
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
    #     else:
    #         train_x = raw_train_x
    #         test_x = raw_test_x

    # for loss_func in ("log", "modified_huber"):
    for loss_func in ("log",):
        # for penalty in ("l1", "l2"):
        for penalty in ("l2",):
            for c in [10 ** (0.5 * j) for j in range(-16, -4)]:
                model = SGDClassifier(loss=loss_func, penalty=penalty, alpha=c, n_iter=1000).fit(train_x, train_y)
                key = ("sgd", use_pca, loss_func, penalty, 2 * numpy.log10(c))
                max_right, least_loss = evaluate(model, key, test_x, test_y, results, max_right, least_loss)
                with open(fname, 'a') as buff:
                    buff.write("{:s},{:.0f},{:d},{:.6f}\n".format(
                        "{:s}_{:s}".format(penalty, loss_func),
                        2 * numpy.log10(c),
                        season,
                        results[key]))
    #
    # # test logistic regressions
    # for penalty in ("l2", "l1"):
    #     for c in [10 ** (0.5 * j) for j in range(0, 10)]:
    #         model = LogisticRegression(penalty=penalty, C=c).fit(train_x, train_y)
    #         key = (penalty, use_pca, 2 * numpy.log10(c))
    #         max_right, least_loss = evaluate(model, key, test_x, test_y, results, max_right, least_loss)
    #         with open(fname, 'a') as buff:
    #             buff.write("{:s},{:.0f},{:d},{:.6f}\n".format(penalty, 2 * numpy.log10(c), season, results[key]))
    #
    #         model = BaggingClassifier(LogisticRegression(penalty=penalty, C=c), n_estimators=100,
    #                                   max_samples=0.5).fit(train_x, train_y)
    #         key = (penalty + " bagging", use_pca, 2 * numpy.log10(c))
    #         max_right, least_loss = evaluate(model, key, test_x, test_y, results, max_right, least_loss)
    best_model = min(results.items(), key=lambda j: numpy.median(j[1]))
    TourneyFeatures(season).write_predictions(best_model, normalizer.transform)
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


def cross_validate():
    results = defaultdict(list)
    test_years = range(2011, 2015)

    fname = os.path.join(OUT_DIR, 'cross_validation.csv')
    with open(fname, 'w') as buff:
        buff.write("penalty,c,season,loss\n")

    for year in test_years:
        print(year)
        year_results = find_best_model(year, fname)
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
    season = 2014
    print(TourneyFeatures(season).score_predictions())
    # raw_train_x, train_y = features_labels(season)
    # normalizer = Normalizer()
    # train_x = normalizer.fit_transform(raw_train_x)
    # model = SGDClassifier(loss='log', penalty='l2', alpha=10 ** -6, n_iter=1000).fit(train_x, train_y)
    # TourneyFeatures(season).write_predictions(model, normalizer.transform)


def main():
    cross_validate()


if __name__ == '__main__':
    main()
