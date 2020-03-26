import os
import json
import numpy
import string
import random
import subprocess
from datetime import datetime as dt
from shutil import rmtree
from pickle import dump
from yaml import safe_load
from pandas import read_csv, DataFrame, concat
from pandas.api.types import CategoricalDtype


INFORMATION_OPTIONAL_KEYS = ["STRATIFICATION_COLUMN", "CATEGORICAL_COLUMN", "ORDINAL_COLUMN", "USELESS_COLUMN"]


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
        self.metadata = None
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

    def retrieve_metadata(self):
        # TODO: make more robust
        model_path_shards = self.model_path.split("/")
        if model_path_shards[0] == 'gs:':
            metadata_path = '/'.join(model_path_shards[0:6] + ['METADATA'] + ['TRAIN'] + ['metadata.json'])
        else:
            metadata_path = self.model_path + '/metadata.json'
        os.system(' '.join(['gsutil ', 'cp', metadata_path, self.local_path]))  # fails if called by subprocess

    def check_info(self):

        required_keys = ["ID_COLUMN", "TARGET_COLUMN"]

        # Required keys
        for key in required_keys:
            if key not in self.info:
                raise KeyError("You must provide {} in YAML information file".format(key))

        # Data Types - Cannot use generic parameters with isinstance
        if not isinstance(self.info["ID_COLUMN"], str):
            raise TypeError("ID_COLUMN must be a string")
        if not isinstance(self.info["TARGET_COLUMN"], str):
            raise TypeError("TARGET_COLUMN must be a string")
        if "USELESS_COLUMN" in self.info:  # list of strings
            if not isinstance(self.info["USELESS_COLUMN"], list):
                raise TypeError("USELESS_COLUMN must be a list")
            for item in self.info["USELESS_COLUMN"]:
                if not isinstance(item, str):
                    raise TypeError("All elements of USELESS_COLUMN must be strings")
        if "STRATIFICATION_COLUMN" in self.info:  # list of strings
            if not isinstance(self.info["STRATIFICATION_COLUMN"], list):
                raise TypeError("STRATIFICATION_COLUMN must be a list")
            for item in self.info["STRATIFICATION_COLUMN"]:
                if not isinstance(item, str):
                    raise TypeError("All elements of STRATIFICATION_COLUMN must be strings")

    def read_info(self, fill=False):
        try:
            file_list = [item for item in os.listdir(self.local_path)
                         if os.path.isfile(os.path.join(self.local_path, item)) and
                         (item.split(".")[-1] == 'yaml' or item.split(".")[-1] == 'yml')]
            if len(file_list) > 1:
                raise ValueError("Found multiple YAML files. You must provide a single YAML.")
            local_info_path = os.path.join(self.local_path, file_list[0])
            with open(local_info_path, 'r') as stream:
                self.info = safe_load(stream)
            self.check_info()
            if fill:
                # Add all optional keys
                for key in INFORMATION_OPTIONAL_KEYS:
                    if key not in self.info:
                        self.info[key] = []
        except:
            rmtree(self.local_path)
            raise Exception("Unable to load info YAML.")

    def read_metadata(self):
        file_list = [item for item in os.listdir(self.local_path)
                     if os.path.isfile(os.path.join(self.local_path, item)) and
                     (item.split(".")[-1] == 'json')]
        if len(file_list) > 1:
            raise ValueError("Found multiple JSON files. You must provide a single JSON.")
        local_info_path = os.path.join(self.local_path, file_list[0])
        with open(local_info_path, 'r') as stream:
            self.metadata = safe_load(stream)

    def _encode_categories_to_integers(self):
        for col in self.data.columns:
            if self.data[col].dtype.name == 'category':
                self.data[col] = self.data[col].cat.codes

    def _generate_data_type_dict(self):
        try:
            d = {}
            for key in self.metadata['column_data_types']:
                if self.metadata['column_data_types'][key] == 'category':
                    d[key] = CategoricalDtype(self.metadata['category_encodings'][key], ordered=False)
                elif self.metadata['column_data_types'][key] == 'float64':
                    d[key] = numpy.float64
            return d
        except (TypeError, KeyError):
            return None

    def read_data(self, encode_cat_to_int=False):
        try:
            file_list = [item for item in os.listdir(self.local_path)
                         if os.path.isfile(os.path.join(self.local_path, item)) and item.split(".")[-1] == 'csv']
            d = self._generate_data_type_dict()
            dfs = []
            for file in file_list:
                if "USELESS_COLUMN" in self.info:
                    dfs.append(read_csv(os.path.join(self.local_path, file),
                                        usecols=lambda w: w not in self.info["USELESS_COLUMN"], dtype=d))
                else:
                    dfs.append(read_csv(os.path.join(self.local_path, file), dtype=d))
            self.data = concat(dfs, axis=0)
            self.data.set_index(self.info["ID_COLUMN"], inplace=True, drop=True)
            # TODO: directly read ordered columns!
            # self.data.reindex(columns=sorted(self.data.columns), copy=False)  # this creates a copy
            if encode_cat_to_int:
                self._encode_categories_to_integers()
        except:
            raise Exception("Unable to load data csv.")

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
