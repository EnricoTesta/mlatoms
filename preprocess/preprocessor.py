import os
import numpy as np
import string
import random
import subprocess
from datetime import datetime as dt
from shutil import rmtree
from pickle import dump
from yaml import safe_load
from pandas import read_csv, DataFrame, Series, concat


class Encoder:
    """How do I know what trials I'm in??? It would be of great help in using HyperTune and setting model name.
    --> Can be found in TrainingOutput..."""

    def __init__(self, train_data_path=None, model_path=None, algo=None, params=None):
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

        # Output properties
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
    def predict(estimator, x):
        try:
            return estimator.predict_proba(x)
        except:
            return estimator.predict(x)

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

        # Step 4 - Fit model on entire train data
        self.trained_model = self.fit(train_data, None)
        self.predictions = self.trained_model.transform(train_data).reset_index()  # TODO: make a universal transform function

        # Step 5 - Export Results
        unique_id = self.generate_unique_id()
        model_file_name = 'preprocess_' + unique_id + '.pkl'
        predictions_file_name = 'data_' + unique_id + '.csv'

        # Export Trained Model
        tmp_model_file = os.path.join('/tmp/', model_file_name)
        with open(tmp_model_file, 'wb') as f:
            # joblib.dump(self.trained_model, f)
            dump(self.trained_model, f)
        subprocess.check_call(['gsutil', 'cp', tmp_model_file, os.path.join(self.model_path, model_file_name)])

        # Export data
        if self.predictions is None:
            raise(TypeError, "Unable to find transformed data")
        else:
            tmp_predictions_file = os.path.join('/tmp/', predictions_file_name)
            self.predictions.to_csv(tmp_predictions_file, index=False)
            subprocess.check_call(
                ['gsutil', 'cp', tmp_predictions_file, os.path.join(self.model_path, predictions_file_name)])

        # Step 6 - Clean-up
        rmtree(local_path)
