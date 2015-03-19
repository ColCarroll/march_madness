import cPickle
import csv
import glob
import json
import os
import re
import itertools
import numpy
import scipy
from sklearn.decomposition.pca import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from data import DataHandler
from league import League

FIRST_SEASON = 2003
DIR = os.path.dirname(os.path.realpath(__file__))
PICKLE_DIR = os.path.join(DIR, "pickles")
OUT_DIR = os.path.join(DIR, 'out_data')
JSON_DIR = os.path.join(DIR, 'model_json')

for directory in (PICKLE_DIR, OUT_DIR, JSON_DIR):
    if not os.path.exists(directory):
        os.mkdir(directory)


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
            team_one_features = max([(k, v) for k, v in self.team_one.features.items() if k[0] == self.season])[1]
            team_two_features = max([(k, v) for k, v in self.team_two.features.items() if k[0] == self.season])[1]
            return team_one_features + team_two_features

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
            for team_two_id in team_ids[j + 1:]:
                team_one = self.league.data(team_one_id)
                team_two = self.league.data(team_two_id)
                game_features = team_features(team_one, team_two, self.season)
                pagerank_one = self.league.strength(team_one_id, self.season)
                pagerank_two = self.league.strength(team_two_id, self.season)
                line = self.league.pointspread(self.season, team_one_id, team_two_id)
                features.append(game_features + [pagerank_one, pagerank_two, line])
                ids.append("{:d}_{:d}_{:d}".format(self.season, team_one_id, team_two_id))
        return numpy.array(features), ids

    def write_predictions(self, model):
        if not os.path.exists(self.pred_dir):
            os.mkdir(self.pred_dir)

        raw_train_x, train_y = features_labels(self.season + 1)
        scaler = StandardScaler()

        train_x = scaler.fit_transform(raw_train_x)
        pca = PCA()
        if model.json.get("use_pca", False):
            train_x = pca.fit_transform(train_x)

        clf = model.func(**model.best_params()["params"]).fit(train_x, train_y)

        features, ids = self.get_features_and_ids()

        features = scaler.transform(features)
        if model.json.get("use_pca", False):
            features = pca.transform(features)

        predictions = clf.predict_proba(features)
        if len(predictions.shape) == 2:
            predictions = predictions[:, 1]

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
            cur.execute("SELECT season, wteam, lteam FROM tourney_compact_results WHERE season=?", (self.season,))
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
                game_features = team_features(wteam, lteam, season, row['daynum'])
                if game_features:
                    w_pagerank = self.league.strength(wteam.id, season, row['daynum'])
                    l_pagerank = self.league.strength(lteam.id, season, row['daynum'])
                    line = self.league.pointspread(season, wteam.id, lteam.id, row['daynum'])
                    features.append(game_features + [w_pagerank, l_pagerank, line])
                    labels.append(1)
                    features.append(team_features(lteam, wteam, season, row['daynum']) +
                                    [l_pagerank, w_pagerank, -line])
                    labels.append(0)
            cPickle.dump(features, open(feature_pickle, 'w'))
            cPickle.dump(labels, open(label_pickle, 'w'))
        return features, labels

    @staticmethod
    def clean():
        map(os.remove, glob.glob(os.path.join(PICKLE_DIR, "*")))


def team_features(team_one, team_two, season, daynum=None):
    game_features = Features(season, team_one, team_two, daynum).features()
    if game_features:
        return game_features


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


class TunedModel:
    def __init__(self, name, func, **base_params):
        self.name = name
        self.func = func
        self.params = base_params
        self._json_fname = os.path.join(JSON_DIR, "{:s}.json".format(name)).replace(" ", "_")
        self._json = None

    @property
    def json(self):
        if self._json is None:
            if not os.path.exists(self._json_fname):
                self._json = {"tests": []}
                self._write()
            self._json = json.load(open(self._json_fname))
        return self._json

    def _write(self):
        json.dump(self._json, open(self._json_fname, 'w'))

    def have_tested(self, params, pca):
        for test in self.json.get("tests", []):
            if sorted(params) == sorted(test["params"].items()) and test.get("use_pca", False) == pca:
                return True
        return False

    def cross_validate(self, train_x, train_y, test_x, test_y, **params):
        if not params:
            params = {"dummy": [0]}
        keys, values = zip(*params.items())
        for param_list in itertools.product(*values):
            cv_params = self.params.items() + zip(keys, param_list)
            for use_pca in (False, True):
                if self.have_tested(cv_params, use_pca):
                    continue
                if use_pca:
                    pca = PCA(n_components=0.99)
                    proc_train_x = pca.fit_transform(train_x)
                    proc_test_x = pca.transform(test_x)
                else:
                    proc_train_x = train_x
                    proc_test_x = test_x
                if "dummy" in params:
                    model = self.func().fit(proc_train_x, train_y)
                else:
                    model = self.func(**dict(cv_params)).fit(proc_train_x, train_y)
                predictions = model.predict_proba(proc_test_x)
                if len(predictions.shape) == 2:
                    predictions = predictions[:, 1]
                num_right = (test_y == predictions.round()).sum()
                self.json["tests"].append({})
                test_data = self.json["tests"][-1]
                test_data["use_pca"] = use_pca
                test_data["pct_right"] = 100 * num_right / float(len(test_y))
                test_data["loss"] = log_loss(test_y, predictions)
                test_data["num_right"] = num_right
                test_data["num_tests"] = len(test_y)
                test_data["params"] = dict(cv_params)
                self._write()
                print(self.print_test(test_data))

    def __repr__(self):
        return "Tuned {:s} model".format(self.name)

    def __str__(self):
        best_test = self.best_params()
        return "{:s}\n{:,d} cross validations".format(
            self.print_test(best_test),
            len(self.json["tests"]),
        )

    def print_test(self, test):
        params = ", ".join(["{:s} = {:s}".format(str(key), str(value)) for key, value in test["params"].items()])
        if test.get("use_pca", False):
            params += ", and with pca compression"
        return "Tuned {:s} model with {:s}\n\tLoss:\t{:.5f}\n\tNum right:\t{:,d} out of {:,d} ({:.2f}%)".format(
            self.name,
            params,
            test["loss"],
            test["num_right"],
            test["num_tests"],
            test["pct_right"]
        )

    def best_params(self):
        if not self.json.get("tests"):
            return self.params
        optimal_key = "loss"
        return min(self.json.get("tests"), key=lambda j: j[optimal_key])

    @staticmethod
    def clean():
        map(os.remove, glob.glob(os.path.join(JSON_DIR, "*")))

    @property
    def model(self):
        params = self.best_params()["params"]
        if "dummy" in params:
            return self.func()
        return self.func(**params)


def season_models(season):
    names = ["Nearest Neighbors", "Logistic Regression",
             "Random Forest",
             "Naive Bayes"]
    classifiers = [
        KNeighborsClassifier,
        LogisticRegression,
        RandomForestClassifier,
        GaussianNB,
    ]

    models = {name: TunedModel("{:s} {:d}".format(name, season), classifier) for
              name, classifier in zip(names, classifiers)}

    def ensemble_builder(**params):
        filtered_models = {k: v for k, v in models.iteritems() if "Ensemble" not in k}
        return EnsembleModel(filtered_models, **params)

    models["Ensemble"] = TunedModel("Ensemble {:d}".format(season), ensemble_builder)
    return models


def cross_validate(season):
    models = season_models(season)
    raw_train_x, train_y = features_labels(season)
    raw_test_x, test_y = map(numpy.array, AllFeatures().features_and_labels(season))
    scaler = StandardScaler()

    train_x = scaler.fit_transform(raw_train_x)
    test_x = scaler.transform(raw_test_x)

    models["Nearest Neighbors"].cross_validate(train_x, train_y, test_x, test_y,
                                               n_neighbors=[100, 200, 500],
                                               weights=['uniform', 'distance'])
    models["Logistic Regression"].cross_validate(train_x, train_y, test_x, test_y,
                                                 C=[10 ** (0.5 * j) for j in range(-16, 16)],
                                                 penalty=["l1", "l2"])
    models["Random Forest"].cross_validate(train_x, train_y, test_x, test_y,
                                           n_estimators=[100, 200, 300, 400, 500],
                                           max_depth=[10, 20, 30, 50, None])
    models["Naive Bayes"].cross_validate(train_x, train_y, test_x, test_y)
    # models["adaboost"].cross_validate(train_x, train_y, test_x, test_y,
    # n_estimators=[100, 200],
    # learning_rate=[10 ** (0.5 * j) for j in range(-16, 0)])
    # models["QDA"].cross_validate(train_x, train_y, test_x, test_y,
    #                              reg_param=[10 ** (0.5 * j) for j in range(-8, 8)])

    models["Ensemble"].cross_validate(train_x, train_y, test_x, test_y, blend=["mean", "median"])

    best_model = min(models.itervalues(), key=lambda j: j.best_params()["loss"])

    # print("Best model: ")
    # print(best_model)
    # return models["Ensemble"]
    return best_model


class EnsembleModel:
    def __init__(self, models, **params):
        self.models = models.values()
        self.model_funcs = [j.model for j in models.values()]
        self.params = params
        self._pca = PCA(n_components=0.99)
        self._clf = None

    def fit(self, x, y):
        train_x, test_x, train_y, test_y, = train_test_split(x, y, test_size=0.2)
        pca_train_x = self._pca.fit_transform(train_x)
        pca_test_x = self._pca.transform(test_x)
        for model, model_func in zip(self.models, self.model_funcs):
            if model.json.get("use_pca", False):
                train_x = pca_train_x
                test_x = pca_test_x
            else:
                pass
            model_func.fit(train_x, train_y)
        self._fit_meta_estimator(test_x, test_y)
        return self

    def _fit_meta_estimator(self, x, y):
        predictions = self._predictions(x).T
        y = numpy.atleast_2d(y).T
        labels = numpy.argmin(abs(predictions - y * numpy.ones((1, predictions.shape[1]))), 1)
        self._clf = GaussianNB().fit(x, labels)

    def _predictions(self, x):
        pca_x = self._pca.transform(x)
        predictions = []
        weights = []

        for model, model_func in zip(self.models, self.model_funcs):
            if model.json.get("use_pca", False):
                test_x = pca_x
            else:
                test_x = x
            predictions.append(model_func.predict_proba(test_x)[:, 1])
            weights.append(model.best_params()["loss"])
        return numpy.array(predictions)

    def predict_proba(self, x):
        blend = self.params.get("blend", "mean")
        predictions = self._predictions(x)
        if blend == "median":
            return numpy.median(predictions, 0)
        if blend == "meta":
            probs = self._clf.predict_proba(x)
            preds = []
            for row, prob in zip(predictions.T, probs):
                if max(prob) > 0.99:
                    preds.append(row[numpy.argmax(prob)])
                else:
                    preds.append(numpy.median(row))
            return numpy.array(preds)

        return predictions.mean(0)


def clean():
    AllFeatures.clean()
    TunedModel.clean()


def main():
    for season in range(2014, 2016):
        best_model = cross_validate(season)
        print("Best Model in {:d}:".format(season))
        print(best_model)
        TourneyFeatures(season).write_predictions(best_model)
        print TourneyFeatures(season).score_predictions()


if __name__ == '__main__':
    main()