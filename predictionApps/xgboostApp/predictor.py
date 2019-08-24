import os
import pickle
import numpy as np
from pandas import DataFrame
from logging import getLogger

logger = getLogger("xgb_predictor")


class MyPredictor(object):
    def __init__(self, model, preprocessor):
        self._model = model
        self._preprocessor = preprocessor

    def predict(self, instances, **kwargs):
        logger.info("Retrieving header, inputs, and info...")
        header = instances[0]  # first row is header row
        inputs = DataFrame(data=instances[1:], columns=header)
        info = kwargs.get('info')

        logger.info("Dropping useless columns. Fetching ids...")
        for col in info["USELESS_COLUMN"] + info["STRATIFICATION_COLUMN"]:
            try:
                inputs.drop(col, inplace=True, axis=1)
            except KeyError:
                logger.warning("Tried to drop \"{}\" but it wasn't found in axis.".format(col))
        ids = inputs.pop(info["ID_COLUMN"])
        raw_data = np.asarray(inputs)
        try:
            preprocessed_inputs = self._preprocessor.preprocess(raw_data)
        except:
            logger.info("No preprocessing applied")
            preprocessed_inputs = raw_data

        # Feature name validation shut off. Expected same order as in train data.
        if kwargs.get('probabilities'):
            logger.info("Predicting probabilities...")
            probabilities = self._model.predict_proba(preprocessed_inputs, validate_features=False)
            return np.concatenate((ids.values.reshape(-1, 1), probabilities), axis=1).tolist()
        else:
            logger.info("Predicting values...")
            outputs = self._model.predict(preprocessed_inputs, validate_features=False)
            return np.concatenate((ids.values.reshape(-1, 1), outputs), axis=1).tolist()

    @classmethod
    def from_path(cls, model_dir):
        # Assumption: there is only one model file in each directory/blob
        model_file_list = [f for f in os.listdir(model_dir)
                           if os.path.isfile(os.path.join(model_dir, f)) and f[0:5] == 'model']
        if len(model_file_list) > 1:
            raise ValueError("Multiple model files found")
        try:
            model_file = model_file_list[0]
        except IndexError:
            raise ValueError("Model file not found")
        model_path = os.path.join(model_dir, model_file)
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except:
            raise ValueError("Unable to unpickle model file")

        # Assumption: there is only one preprocess file in each directory/blob
        preprocessor_file_list = [f for f in os.listdir(model_dir)
                                  if os.path.isfile(os.path.join(model_dir, f)) and f[0:12] == 'preprocessor']
        if len(preprocessor_file_list) > 1:
            raise ValueError("Multiple preprocessor files found")
        try:
            preprocessor_file = preprocessor_file_list[0]
        except IndexError:
            print("No preprocessor file found")
            preprocessor_file = ""
        preprocessor_path = os.path.join(model_dir, preprocessor_file)
        try:
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
        except:
            preprocessor = None

        return cls(model, preprocessor)
