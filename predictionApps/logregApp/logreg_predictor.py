import numpy as np
from ..predictor import BasePredictor


class MyPredictor(BasePredictor):

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
