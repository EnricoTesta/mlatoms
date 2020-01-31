import os
import json
import string
import random
import subprocess
from datetime import datetime as dt
from shutil import rmtree
from pickle import dump
from yaml import safe_load
from pandas import read_csv, DataFrame, concat


class Atom:

    def __init__(self, data_path=None, model_path=None, algo=None, params=None):
        # Input information
        if data_path is None:
            raise ValueError("Must set data_path")

        self.data_path = data_path
        self.model_path = model_path
        self.algo = algo  # this is a class
        if params is None:
            self.params = {'algo': {}, 'fit': {}}
        else:
            self.params = params

        # Aux properties
        self.local_path = None
        self.info = None
        self.data = None

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

    def fit(self, x, y):
        return self.algo(**self.params['algo']).fit(x, y)

    @staticmethod
    def transform(estimator, x):
        return estimator.transform(x)

    @staticmethod
    def predict(estimator, x):
        try:
            return estimator.predict_proba(x)
        except:
            return estimator.predict(x)

    def make_local_path(self):
        self.local_path = os.getcwd() + "/tmp/"
        if os.path.exists(self.local_path):  # start from scratch
            rmtree(self.local_path)
        os.makedirs(self.local_path)

    def retrieve_data(self):
        self.make_local_path()
        os.system(' '.join(['gsutil -m', 'rsync -r', self.data_path, self.local_path]))  # fails if called by subprocess

    def read_info(self):
        try:
            local_info_path = os.path.join(self.local_path, "info.yml")
            with open(local_info_path, 'r') as stream:
                self.info = safe_load(stream)
        except:
            rmtree(self.local_path)
            raise Exception("Unable to load info file.")

    def read_data(self):
        try:
            file_list = [item for item in os.listdir(self.local_path)
                         if os.path.isfile(os.path.join(self.local_path, item)) and item.split(".")[-1] == 'csv']
            if len(file_list) == 1:
                self.data = read_csv(os.path.join(self.local_path, file_list[0]),
                                     usecols=lambda w: w not in self.info["USELESS_COLUMN"])
            else:
                dfs = []
                for file in file_list:
                    dfs.append(read_csv(os.path.join(self.local_path, file),
                               usecols=lambda w: w not in self.info["USELESS_COLUMN"]))
                self.data = concat(dfs, axis=0)
            self.data.set_index(self.info["ID_COLUMN"], inplace=True, drop=True)
        except:
            raise Exception("Unable to load data file.")

    def export_json(self, json_object, json_file_name):
        tmp_json_file = os.path.join(self.local_path, json_file_name)
        with open(tmp_json_file, 'w') as f:
            json.dump(json_object, f)
        subprocess.check_call(['gsutil', 'cp', tmp_json_file, os.path.join(self.model_path, json_file_name)])

    def export_trained_model(self, unique_id):
        model_file_name = 'model_' + unique_id + '.pkl'
        tmp_model_file = os.path.join('/tmp/', model_file_name)
        with open(tmp_model_file, 'wb') as f:
            dump(self.trained_model, f)
        subprocess.check_call(['gsutil', 'cp', tmp_model_file, os.path.join(self.model_path, model_file_name)])

    def export_predictions(self, unique_id):
        predictions_file_name = 'predictions_' + unique_id + '.csv'
        if self.predictions is None:
            pass
        else:
            tmp_predictions_file = os.path.join('/tmp/', predictions_file_name)
            self.predictions.to_csv(tmp_predictions_file, index=False)
            subprocess.check_call(
                ['gsutil', 'cp', tmp_predictions_file, os.path.join(self.model_path, predictions_file_name)])

    def export_validation(self, unique_id):
        validation_file_name = 'info_' + unique_id + '.csv'
        if self.validation is None:
            pass
        else:
            tmp_validation_file = os.path.join('/tmp/', validation_file_name)
            DataFrame.from_dict(self.validation).to_csv(tmp_validation_file, index=False)
            subprocess.check_call(['gsutil', 'cp', tmp_validation_file,
                                   os.path.join(self.model_path, validation_file_name)])

    def run(self):
        """Subclass-specific task"""
        pass
