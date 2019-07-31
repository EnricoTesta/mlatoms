import os
import pickle
import numpy as np
import sklearn


class MyPredictor(object):
    """An example Predictor for an AI Platform custom prediction routine."""

    def __init__(self, model, preprocessor):
        """Stores artifacts for prediction. Only initialized via `from_path`.
        """
        self._model = model
        self._preprocessor = preprocessor

    def predict(self, instances, **kwargs):
        """Performs custom prediction.

        Preprocesses inputs, then performs prediction using the trained Keras
        model.

        Args:
            instances: A list of prediction input instances.
            **kwargs: A dictionary of keyword args provided as additional
                fields on the predict request body.

        Returns:
            A list of outputs containing the prediction results.
        """
        inputs = np.asarray(instances)
        try:
            preprocessed_inputs = self._preprocessor.preprocess(inputs)
        except:
            preprocessed_inputs = inputs
        outputs = self._model.predict(preprocessed_inputs)
        return outputs.tolist()

    @classmethod
    def from_path(cls, model_dir):
        """Creates an instance of MyPredictor using the given path.

        This loads artifacts that have been copied from your model directory in
        Cloud Storage. MyPredictor uses them during prediction.

        Args:
            model_dir: The local directory that contains the trained Keras
                model and the pickled preprocessor instance. These are copied
                from the Cloud Storage model directory you provide when you
                deploy a version resource.

        Returns:
            An instance of `MyPredictor`.
        """
        model_path = os.path.join(model_dir, 'model_20190607_042641_Gnp17Q6Sir.pkl')  # just for testing purposes
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except:
            #tmp = sklearn.linear_model.LogisticRegression()
            #X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
            #model = tmp.fit(X, y)
            model = None

        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        try:
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
        except:
            preprocessor = None

        return cls(model, preprocessor)
