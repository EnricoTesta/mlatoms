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
        default='/mlatoms/data/classification/binary/ieee-fraud/score/',
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
                                                  'encode_target_to_int': False, 'encode_target_to_one_hot': False}}
    for item in args_dict:
        if item == 'use_proba':
            try:
                param_dict[item] = True if args_dict[item] > 0 else param_dict[item] = False
            except TypeError:
                param_dict[item] = True if args_dict[item] == "1" else param_dict[item] = False
        param_dict[item] = args_dict[item]

    t = LGBMBatchPredictor(data_path=param_dict['score_dir'], model_path=param_dict['model_file'], params=param_dict,
                           preprocess_path=None, use_probabilities=param_dict['use_proba'],
                           output_dir=param_dict['output_dir'])
    t.run()


if __name__ == '__main__':
    main()
