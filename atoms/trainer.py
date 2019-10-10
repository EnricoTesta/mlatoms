"""Defines the trainer class template that is then subclassed in all mlatoms.

- Should expect train data location
- Should expect model location (where to save outputs)
- Should expect all relevant arguments for hypertune
- Should return a trained model
- Should return a validation/CV/test assessment
- Should return an out-of-fold prediction dataset (for staking purposes)

For the moment exclude Spark/Tensorflow models. Those will use standard APIs anyway. Suppose to use pandas processing
and suppose you are dealing with small data (max 1 GB).

"""
import os
import numpy as np
import string
import random
import subprocess
import hypertune
from datetime import datetime as dt
from shutil import rmtree
from pickle import dump
from yaml import safe_load
from pandas import read_csv, DataFrame, Series, concat
from sklearn.model_selection import StratifiedKFold, KFold, PredefinedSplit, cross_validate
from sklearn import metrics


CLASSIFICATION_ESTIMATORS = ['LogisticRegression', 'LGBMClassifier',
                             'XGBClassifier', 'AutoSklearnClassifier', 'DummyClassifier']
REGRESSION_ESTIMATORS = ['LinearRegression']
BENCHMARK_ESTIMATORS = ['DummyClassifier']
CV_FOLDS = 3
HYPERTUNE_LOSSES = {'binary_crossentropy': 'log_loss',
                    'categorical_crossentropy': 'log_loss',
                    'accuracy': 'accuracy'}


class Trainer:
    """How do I know what trials I'm in??? It would be of great help in using HyperTune and setting model name.
    --> Can be found in TrainingOutput..."""

    def __init__(self, train_data_path=None, model_path=None, algo=None, params=None, hypertune_loss=None):
        # Input information
        if train_data_path is None or model_path is None:
            raise ValueError("Must set train_data_path and model_path")

        self.train_data_path = train_data_path
        self.model_path = model_path
        self.algo = algo  # this is a class
        self.hypertune_loss = hypertune_loss
        if params is None:
            self.params = {'algo': {}, 'fit': {}}
        else:
            self.params = params

        # Output properties
        self.problem_specs = None
        self.validation = None
        self.predictions = None
        self.trained_model = None

    @staticmethod
    def generate_unique_id():
        available_characters = string.ascii_letters + string.digits
        ts = dt.strftime(dt.now(), "%Y%m%d_%H%M%S") + "_"
        suffix = ''.join((random.choice(available_characters) for _ in range(10)))
        return ts + suffix

    def _get_scoring_list(self, target):
        scoring_dict = {}
        if self.algo.__name__ in CLASSIFICATION_ESTIMATORS:

            scoring_dict['accuracy'] = metrics.make_scorer(metrics.accuracy_score)
            scoring_dict['log_loss'] = metrics.make_scorer(metrics.log_loss, greater_is_better=False, needs_proba=True)
            scoring_dict['matthews_corr'] = metrics.make_scorer(metrics.matthews_corrcoef)
            if target.unique().shape[0] == 2:  # binary
                scoring_dict['roc_auc'] = metrics.make_scorer(metrics.roc_auc_score)  # average: 'macro' is default
                scoring_dict['hinge_loss'] = metrics.make_scorer(metrics.hinge_loss, greater_is_better=False)
                scoring_dict['f1'] = metrics.make_scorer(metrics.f1_score)
                scoring_dict['precision'] = metrics.make_scorer(metrics.precision_score)
                scoring_dict['recall'] = metrics.make_scorer(metrics.recall_score)
                scoring_dict['fbeta'] = metrics.make_scorer(metrics.fbeta_score, beta=1, average='binary')
            else:  # multi-class
                scoring_dict['f1'] = metrics.make_scorer(metrics.f1_score, average='macro')
                scoring_dict['precision'] = metrics.make_scorer(metrics.precision_score, average='macro')
                scoring_dict['recall'] = metrics.make_scorer(metrics.recall_score, average='macro')
                scoring_dict['fbeta'] = metrics.make_scorer(metrics.fbeta_score, beta=1, average='macro')

        elif self.algo.__name__ in REGRESSION_ESTIMATORS:
            raise NotImplementedError
        else:
            raise NotImplementedError
        return scoring_dict

    def _get_score_method(self):
        if self.algo.__name__ in CLASSIFICATION_ESTIMATORS:
            return 'predict_proba'
        elif self.algo.__name__ in REGRESSION_ESTIMATORS:
            return 'predict'
        else:
            raise NotImplementedError

    def _get_estimator_type(self):
        if self.algo.__name__ in CLASSIFICATION_ESTIMATORS:
            return 'classification'
        elif self.algo.__name__ in REGRESSION_ESTIMATORS:
            return 'regression'
        else:
            raise NotImplementedError

    def fit(self, x, y):
        return self.algo(**self.params['algo']).fit(x, y)

    @staticmethod
    def predict(estimator, x):
        try:
            return estimator.predict_proba(x)
        except:
            return estimator.predict(x)

    @staticmethod
    def infer_problem_specs(x, y):
        """Returns a dictionary with 'type': regression/classification, 'balanced': True/False/None,
         'binary': True/False/None, 'n_dimensions': integer, 'n_samples': integer"""
        d = {'n_dimensions': x.shape[1], 'n_samples': x.shape[0]}
        yu, yu_counts = np.unique(y, return_counts=True)
        if len(yu) >= 100 or np.count_nonzero(yu - np.around(yu)) > 0:
            d['type'] = 'regression'
            d['binary'] = None
            d['balanced'] = None
        else:
            d['type'] = 'classification'
        if d['type'] == 'classification':
            d['binary'] = len(yu) == 2
            d['balanced'] = bool(yu_counts.max() == yu_counts.min())
        return d

    def generate_folds(self, x, y, stratification=None):
        # TODO: manage shuffle without messing up with original DataFrame indexes
        test_folds = -np.ones(y.shape)
        counter = 0
        if stratification is None:
            for _, test_index in KFold(n_splits=CV_FOLDS, shuffle=False).split(x):
                test_folds[test_index] = counter
                counter += 1
        else:
            # Build stratification variable
            tmp_var = stratification.astype(str)
            tmp_var['target'] = y.astype(str)
            for column in tmp_var.columns:
                try:
                    strat_var += tmp_var[column]
                except NameError:
                    strat_var = tmp_var[column]
            # Draw folds
            for _, test_index in StratifiedKFold(n_splits=CV_FOLDS, shuffle=False).split(x, strat_var):
                test_folds[test_index] = counter
                counter += 1
        return PredefinedSplit(test_folds)

    def generalization_assessment(self, x, y, cv):
        validation = cross_validate(self.algo(**self.params['algo']), x, y=y, scoring=self._get_scoring_list(y),
                                    return_train_score=True, return_estimator=True, n_jobs=-1, cv=cv)
        estimators = validation.pop('estimator')

        # Get out-of-fold predictions
        pred = concat([DataFrame(self.predict(fit_model, x.loc[x.index[cv.test_fold == idx], :]),
                                 index=x.index[cv.test_fold == idx])
                      for idx, fit_model in enumerate(estimators)], axis=0)
        predictions_col_names = []
        if pred.shape[1] == 1:
            predictions_col_names.append("value")  # either class labels or regression values
        else:
            for i in range(pred.shape[1]):
                predictions_col_names.append("probability_" + str(i))
        pred.columns = predictions_col_names
        pred.reset_index(inplace=True)

        # Add info to validation
        validation['benchmark'] = int(self.algo.__name__ in BENCHMARK_ESTIMATORS)
        # Logloss correction
        validation['train_log_loss'] = -validation['train_log_loss']
        validation['test_log_loss'] = -validation['test_log_loss']
        # Algo name
        validation['algo'] = self.algo.__name__
        return validation, pred

    def run(self):

        # Step 0 - Retrieve data folder
        local_path = os.getcwd() + "/tmp/"
        if os.path.exists(local_path):  # start from scratch
            rmtree(local_path)
        os.makedirs(local_path)
        os.system(' '.join(['gsutil -m', 'rsync', self.train_data_path, local_path]))  # fails if called by subprocess

        # Step 1 - Read info.yml
        try:
            local_info_path = os.path.join(local_path, "info.yml")
            with open(local_info_path, 'r') as stream:
                info = safe_load(stream)
        except:
            rmtree(local_path)
            raise Exception("Unable to load info file.")

        # Step 1 - Read data (csv format)
        try:
            file_list = [item for item in os.listdir(local_path)
                         if os.path.isfile(os.path.join(local_path, item)) and item.split(".")[-1] == 'csv']
            if len(file_list) == 1:
                train_data = read_csv(os.path.join(local_path, file_list[0]),
                                      usecols=lambda w: w not in info["USELESS_COLUMN"])
            else:
                dfs = []
                for file in file_list:
                    dfs.append(read_csv(os.path.join(local_path, file),
                                        usecols=lambda w: w not in info["USELESS_COLUMN"]))
                train_data = concat(dfs, axis=0)
            train_data.set_index(info["ID_COLUMN"], inplace=True, drop=True)
            idx = Series(train_data.index, name="id")
        except:
            raise Exception("Unable to load train data file.")
        y = train_data[info["TARGET_COLUMN"]]
        if self.algo.__name__ in CLASSIFICATION_ESTIMATORS and y.apply(lambda x: x - int(x) != 0).any():  # THIS IS SLOW
            raise ValueError("Target variable for classification algorithms must be integer encoded.")
        x = train_data.iloc[:, np.isin(train_data.columns, [info["TARGET_COLUMN"]] + info["STRATIFICATION_COLUMN"],
                                       invert=True)]

        self.problem_specs = self.infer_problem_specs(x, y)

        # Step 2 - Generalization Assessment
        if info["STRATIFICATION_COLUMN"][0] == '':  # TODO: fix this check with a more robust one
            strat_df = None
        else:
            strat_df = train_data[info["STRATIFICATION_COLUMN"]]
        cv = self.generate_folds(x, y, stratification=strat_df)
        self.validation, self.predictions = self.generalization_assessment(x, y, cv=cv)

        # Step 4 - Fit model on entire train data
        self.trained_model = self.fit(x, y)

        # Step 5 - Export Results
        unique_id = self.generate_unique_id()
        model_file_name = 'model_' + unique_id + '.pkl'
        validation_file_name = 'info_' + unique_id + '.csv'
        predictions_file_name = 'predictions_' + unique_id + '.csv'

        # TODO: implement debug mode for this part

        # Export Trained Model
        tmp_model_file = os.path.join('/tmp/', model_file_name)
        with open(tmp_model_file, 'wb') as f:
            # joblib.dump(self.trained_model, f)
            dump(self.trained_model, f)
        subprocess.check_call(['gsutil', 'cp', tmp_model_file, os.path.join(self.model_path, model_file_name)])

        # Export generalization assessment
        if self.validation is None:
            pass
        else:
            tmp_validation_file = os.path.join('/tmp/', validation_file_name)
            DataFrame.from_dict(self.validation).to_csv(tmp_validation_file, index=False)
            subprocess.check_call(['gsutil', 'cp', tmp_validation_file,
                                   os.path.join(self.model_path, validation_file_name)])

        # Export predictions
        if self.predictions is None:
            pass
        else:
            tmp_predictions_file = os.path.join('/tmp/', predictions_file_name)
            self.predictions.to_csv(tmp_predictions_file, index=False)
            subprocess.check_call(
                ['gsutil', 'cp', tmp_predictions_file, os.path.join(self.model_path, predictions_file_name)])

        # Step 6 - Clean-up
        rmtree(local_path)

        # Step 7 - Report Loss to Hypertune
        if self.hypertune_loss is None or self.validation is None:
            return
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=self.hypertune_loss,
            metric_value=self.validation['test_' + HYPERTUNE_LOSSES[self.hypertune_loss]].mean())
