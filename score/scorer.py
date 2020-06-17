import os
import pickle
import subprocess
import numpy as np
from shutil import rmtree
from pandas import DataFrame
from logging import getLogger
from atoms import Atom

logger = getLogger("sklearn_predictor")


class BatchPredictor(Atom):
    def __init__(self, data_path, model_path, preprocess_path=None,
                 algo=None, params=None, output_dir=None, use_probabilities=False):
        super().__init__(data_path, model_path, algo, params)
        self.preprocess_path = preprocess_path  # this is either a string or a list of strings
        self._model = None
        self._preprocessor = None
        self._output_dir = output_dir
        self._use_probabilities = use_probabilities

    def default_preprocess(self, info, inputs):
        logger.info("Dropping useless columns. Fetching ids...")
        column_to_drop_list = [col for col in self.data.columns if self.info["TARGET_COLUMN"] in col]
        for key in ["USELESS_COLUMN", "STRATIFICATION_COLUMN"]:
            try:
                column_to_drop_list += info[key]
            except KeyError:
                pass
        for col in column_to_drop_list:
            try:
                inputs.drop(col, inplace=True, axis=1)
            except KeyError:
                logger.warning("Tried to drop \"{}\" but it wasn't found in axis.".format(col))
        ids = inputs.index
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

    def retrieve_preprocess(self):
        if self.preprocess_path is not None:
            try:
                if isinstance(self.preprocess_path, list):
                    for item in self.preprocess_path:
                        os.system(' '.join(['gsutil -m', 'cp', item, self.local_path]))
                else:
                    os.system(' '.join(['gsutil -m', 'cp', self.preprocess_path, self.local_path]))
            except:
                raise ValueError("Preprocess file not found")

    def retrieve_model(self):
        # Fetch model & preprocess from GCS
        try:
            os.system(' '.join(['gsutil -m', 'cp', self.model_path, self.local_path]))
        except:
            raise ValueError("Model file not found")

    def restore_preprocess(self):
        # Restore preprocess
        try:
            self._preprocessor = []
            if isinstance(self.preprocess_path, list):
                for item in self.preprocess_path:
                    with open(os.path.join(self.local_path, item), 'rb') as f:
                        self._preprocessor.append(pickle.load(f))  # TODO: ensure preprocess order
            else:
                with open(os.path.join(self.local_path, self.preprocess_path), 'rb') as f:
                    self._preprocessor.append(pickle.load(f))
        except:
            self._preprocessor = None

    def restore_model(self):
        # Restore model
        try:
            with open(os.path.join(self.local_path, self.model_path.split("/")[-1]), 'rb') as f:
                self._model = pickle.load(f)
        except:
            raise ValueError("Unable to unpickle model file")

    def model_predict_proba(self, inputs):
        return inputs

    def model_predict(self, inputs):
        return inputs

    def run(self):

        # Fetch data
        self.retrieve_data()
        self.retrieve_metadata()
        self.read_info()
        self.read_metadata()
        self.read_data(**self.params['read'])

        # Load model
        self.retrieve_preprocess()
        self.retrieve_model()
        self.restore_preprocess()
        self.restore_model()

        # Score model
        ids, raw_data = self.default_preprocess(self.info, self.data)
        preprocessed_inputs = raw_data

        try:
            for item in self._preprocessor:
                preprocessed_inputs = item.preprocess(preprocessed_inputs)
        except:
            logger.info("No preprocessing applied")

        tmp_file_path = os.path.join(self.local_path, 'results.csv')
        if self._use_probabilities:
            logger.info("Predicting probabilities...")
            probabilities = self.model_predict_proba(preprocessed_inputs)
            column_names = self.generate_column_names(probabilities)
            DataFrame(np.concatenate((ids.values.reshape(-1, 1), probabilities), axis=1),
                      columns=['id'] + column_names).to_csv(path_or_buf=tmp_file_path, index=False)
        else:
            logger.info("Predicting values...")
            outputs = self.model_predict(preprocessed_inputs)
            column_names = self.generate_column_names(outputs)
            DataFrame(np.concatenate((ids.values.reshape(-1, 1), outputs), axis=1),
                      columns=['id'] + column_names).to_csv(path_or_buf=tmp_file_path, index=False)

        # self.predictions = self.transform(self.trained_model, self.data).reset_index()  # TODO: verify index

        # Step 5 - Send results to GCS
        subprocess.check_call(['gsutil', 'cp', tmp_file_path,
                               os.path.join(self._output_dir, 'results.csv')])

        # Clean-up
        rmtree(self.local_path)

