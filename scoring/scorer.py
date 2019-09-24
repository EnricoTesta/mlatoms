import os
import pickle
import numpy as np
from shutil import rmtree
from yaml import safe_load
from pandas import DataFrame, read_csv, concat
from logging import getLogger

logger = getLogger("sklearn_predictor")


class BatchPredictor(object):
    def __init__(self, model, preprocessor, output_dir, use_probabilities):
        self._model = model
        self._preprocessor = preprocessor
        self._output_dir = output_dir
        self._use_probabilities = use_probabilities

    @staticmethod
    def make_temporary_directory():
        local_path = os.getcwd() + "/tmp/"
        if os.path.exists(local_path):  # start from scratch
            rmtree(local_path)
        os.makedirs(local_path)
        return local_path

    def default_preprocess(self, info, inputs):
        logger.info("Dropping useless columns. Fetching ids...")
        for col in info["USELESS_COLUMN"] + info["STRATIFICATION_COLUMN"]:
            try:
                inputs.drop(col, inplace=True, axis=1)
            except KeyError:
                logger.warning("Tried to drop \"{}\" but it wasn't found in axis.".format(col))
        ids = inputs.pop(info["ID_COLUMN"])
        raw_data = np.asarray(inputs)
        return ids, raw_data

    @staticmethod
    def generate_column_names(result):
        if result.shape[1] == 1:
            return ['value']  # regression setting
        else:
            names = []
            for i in range(result.shape[1]):
                names.append('probability_' + str(i))  # classification setting
            return names

    def predict(self, score_dir):
        # Assumption: score_dir contains a single info.yml file and multiple csv files

        # Fetch files from score_dir
        local_path = self.make_temporary_directory()
        try:
            os.system(' '.join(['gsutil -m', 'rsync', score_dir, local_path]))
        except:
            raise ValueError("Unable to fetch score data.")

        # Step 1 - Read info.yml
        try:
            local_info_path = os.path.join(local_path, "info.yml")
            with open(local_info_path, 'r') as stream:
                info = safe_load(stream)
        except:
            rmtree(local_path)
            raise Exception("Unable to load info file.")

        # Step 2 - Read score data from csv
        score_data_files = [f for f in os.listdir(local_path)
                            if os.path.isfile(os.path.join(local_path, f)) and f.split(".")[-1] == 'csv']
        try:
            df_list = []
            for f in score_data_files:
                df_list.append(read_csv(filepath_or_buffer=os.path.join(local_path, f)))
            inputs = concat(df_list, axis=0)
        except:
            raise Exception("Unable to read data in memory.")

        ids, raw_data = self.default_preprocess(info, inputs)

        # Step 3 - Preprocess
        try:
            preprocessed_inputs = self._preprocessor.preprocess(raw_data)
        except:
            logger.info("No preprocessing applied")
            preprocessed_inputs = raw_data

        # Step 4 - Predict
        if self._use_probabilities:
            logger.info("Predicting probabilities...")
            probabilities = self._model.predict_proba(preprocessed_inputs)
            column_names = self.generate_column_names(probabilities)
            DataFrame(np.concatenate((ids.values.reshape(-1, 1), probabilities), axis=1),
                      columns=['id'] + column_names).to_csv(path_or_buf=self._output_dir + '/results.csv', index=False)
        else:
            logger.info("Predicting values...")
            outputs = self._model.predict(preprocessed_inputs)
            column_names = self.generate_column_names(outputs)
            DataFrame(np.concatenate((ids.values.reshape(-1, 1), outputs), axis=1),
                      columns=['id'] + column_names).to_csv(path_or_buf=self._output_dir + '/results.csv', index=False)

        # Step 5 - Clean-up
        rmtree(local_path)

    @classmethod
    def from_path(cls, model_file_path, preprocess_file, output_dir, use_probabilities):

        local_path = cls.make_temporary_directory()

        # Fetch model & preprocess from GCS
        try:
            os.system(' '.join(['gsutil -m', 'cp', model_file_path, local_path]))
        except:
            raise ValueError("Model file not found")

        if preprocess_file is not None:
            try:
                os.system(' '.join(['gsutil -m', 'cp', preprocess_file, local_path]))
            except:
                raise ValueError("Preprocess file not found")

        # Restore objects
        model_path = os.path.join(local_path, model_file_path.split("/")[-1])
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except:
            raise ValueError("Unable to unpickle model file")

        try:
            preprocessor_path = os.path.join(local_path, preprocess_file)
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
        except:
            preprocessor = None

        # Clean-up
        rmtree(local_path)

        return cls(model, preprocessor, output_dir, use_probabilities)
