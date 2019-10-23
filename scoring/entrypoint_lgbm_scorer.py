import argparse
import lightgbm
from scorer import BatchPredictor


class LGBMBatchPredictor(BatchPredictor):

    def model_predict(self, preprocessed_input):
        return self._model.predict(preprocessed_input)

    def model_predict_proba(self, preprocessed_input):
        return self._model.predict_proba(preprocessed_input)


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='LGBM batch scorer')
    parser.add_argument(
        '--model-file',
        default=None,
        metavar='model_file',
        help='The absolute path to fetch the model')
    parser.add_argument(
        '--score-dir',
        default='/mlatoms/data/classification/binary/numeric/score/',
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
    param_dict = {}
    for item in args_dict:
        param_dict[item] = args_dict[item]

    t = LGBMBatchPredictor(data_path=param_dict['score_dir'], model_path=param_dict['model_file'],
                           preprocess_path=None, use_probabilities=param_dict['use_proba'],
                           output_dir=param_dict['output_dir'])
    t.run()


if __name__ == '__main__':
    main()
