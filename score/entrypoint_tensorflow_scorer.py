import os
import argparse
import tensorflow as tf
from scorer import BatchPredictor


class TFBatchPredictor(BatchPredictor):

    def retrieve_model(self):
        # Fetch model from GCS
        try:
            if not os.path.isdir(os.path.join(self.local_path, self.model_path.split("/")[-1])):
                os.mkdir(os.path.join(self.local_path, self.model_path.split("/")[-1]))
            os.system(' '.join(['gsutil', 'rsync -r', self.model_path,
                                os.path.join(self.local_path, self.model_path.split("/")[-1])]))
        except:
            raise ValueError("Model folder not found")

    def restore_model(self):
        try:
            self._model = tf.keras.models.load_model(os.path.join(self.local_path, self.model_path.split("/")[-1]))
        except:
            raise ValueError("Unable to unpickle model file")

    def model_predict(self, preprocessed_input):
        return self._model.predict(preprocessed_input)

    def model_predict_proba(self, preprocessed_input):
        return self.model_predict(preprocessed_input)  # predict_proba deprecated in TF2


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='Tensorflow batch scorer')
    parser.add_argument(
        '--model-file',
        default='/mlatoms/test/modeldir/model_20200611_051343_hiFHUYACpn',
        metavar='model_file',
        help='The absolute path to fetch the model')
    parser.add_argument(
        '--score-dir',
        default='/mlatoms/data/classification/multi/numer/score/',
        metavar='score_dir',
        help='The directory to fetch score data')
    parser.add_argument(
        '--output-dir',
        default='/mlatoms/test/outputs/',
        metavar='output_dir',
        help='The directory to store output data')
    parser.add_argument(
        '--use-proba',
        default=1,
        metavar='use_proba',
        help='Flag whether or not to use probabilities.')

    args = parser.parse_args()
    return args


def main():
    # Training settings
    args = get_args()
    args_dict = vars(args)

    # Make param dict
    param_dict = {'read': {'encode_features_to_int': False, 'encode_features_to_one_hot': True,
                                                  'encode_target_to_int': False, 'encode_target_to_one_hot': True}}
    for item in args_dict:
        param_dict[item] = args_dict[item]

    t = TFBatchPredictor(data_path=param_dict['score_dir'], model_path=param_dict['model_file'], params=param_dict,
                         preprocess_path=None, use_probabilities=param_dict['use_proba'],
                         output_dir=param_dict['output_dir'])
    t.run()


if __name__ == '__main__':
    main()
