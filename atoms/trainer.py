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
import string
import random
import subprocess
import hypertune
from datetime import datetime as dt
from shutil import rmtree
from pickle import dump
from yaml import safe_load
from pandas import read_csv, DataFrame, concat
from sklearn.model_selection import cross_validate, cross_val_predict


CLASSIFICATION_ESTIMATORS = ['LogisticRegression', 'LGBMClassifier', 'XGBClassifier', 'AutoSklearnClassifier']
REGRESSION_ESTIMATORS = ['LinearRegression']


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
        if params is None:
            self.params = {'algo': {}, 'fit': {}}
        else:
            self.params = params
        self.hypertune_loss = hypertune_loss

        # Output properties
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
        if self.algo.__name__ in CLASSIFICATION_ESTIMATORS:
            classification_metrics = ['neg_log_loss', 'accuracy']
            if target.unique().shape[0] == 2:
                classification_metrics += ['roc_auc']  # binary case
            return classification_metrics
        elif self.algo.__name__ in REGRESSION_ESTIMATORS:
            return ['neg_median_absolute_error']
        else:
            raise NotImplementedError

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

    def generalization_assessment(self, x, y):
        return cross_validate(self.algo(**self.params['algo']), x, y=y, scoring=self._get_scoring_list(y),
                              return_train_score=True, n_jobs=-1, cv=3)

    def get_out_of_samples_prediction(self, x, y):
        return cross_val_predict(self.algo(**self.params['algo']), x, y=y, n_jobs=-1, cv=3,
                                 method=self._get_score_method())

    def fit(self, x, y):
        return self.algo(**self.params['algo']).fit(x, y)

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
                                      usecols=lambda w: w not in info["USELESS_COLUMN"] + [info["ID_COLUMN"]])
            else:
                dfs = []
                for file in file_list:
                    dfs.append(read_csv(os.path.join(local_path, file),
                                        usecols=lambda w: w not in info["USELESS_COLUMN"] + [info["ID_COLUMN"]]))
                train_data = concat(dfs, axis=0)
        except:
            raise Exception("Unable to load train data file.")
        y = train_data[info["TARGET_COLUMN"]]
        x = train_data.iloc[:, train_data.columns != info["TARGET_COLUMN"]]

        # Step 2 - Cross Validation generalization assessment
        # TODO: define a CV strategy
        self.validation = self.generalization_assessment(x, y)

        # Step 3 - Compute out-of-fold predictions (useful for stacking)
        # TODO: define a CV strategy
        self.predictions = self.get_out_of_samples_prediction(x, y)
        try:
            if self._get_estimator_type() == 'classification':
                self.predictions = self.predictions[:, 1]  # keep only probabilty of class 1
        except TypeError:
            pass  # this happens when subclass overrides prediction method. Should find a better way to handle this

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
            DataFrame(self.predictions).to_csv(tmp_predictions_file, index=False)
            subprocess.check_call(
                ['gsutil', 'cp', tmp_predictions_file, os.path.join(self.model_path, predictions_file_name)])

        # Step 6 - Report Loss to Hypertune
        if self.hypertune_loss is None or self.validation is None:
            return
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=self.hypertune_loss,
            metric_value=self.validation['test_' + self.hypertune_loss].mean())
