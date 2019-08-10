import os
import pickle
import numpy as np


class MyPredictor(object):
    def __init__(self, model, preprocessor):
        self._model = model
        self._preprocessor = preprocessor

    def predict(self, instances, **kwargs):
        inputs = np.asarray(instances)
        try:
            preprocessed_inputs = self._preprocessor.preprocess(inputs)
        except:
            preprocessed_inputs = inputs

        if kwargs.get('probabilities'):
            probabilities = self._model.predict_proba(preprocessed_inputs)
            return probabilities.tolist()
        else:
            outputs = self._model.predict(preprocessed_inputs)
            return outputs.tolist()

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
