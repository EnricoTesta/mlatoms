from trainer import Trainer
from sklearn.dummy import DummyRegressor
import argparse


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='Dummy Regressor')
    parser.add_argument(
        '--strategy',
        type=str,
        default='mean',
        metavar='strategy',
        help="Used to specify a strategy. Can take values: 'mean', 'median', 'quantile', 'constant'")
    parser.add_argument(
        '--constant',
        type=float,
        default=None,
        metavar='constant',
        help="Useful only with the constant strategy")
    parser.add_argument(
        '--model-dir',
        default='/mlatoms/test/modeldir',
        metavar='model_dir',
        help='The directory to store the model')
    parser.add_argument(
        '--train-files',
        default='/mlatoms/data/regression/numer/',
        metavar='train_files',
        help='The directory to fetch train data')

    args = parser.parse_args()
    return args


def main():
    # Training settings
    args = get_args()
    args_dict = vars(args)

    # Make param dict
    param_dict = {'algo': {}, 'fit': {}, 'read': {'encode_features_to_int': False, 'encode_features_to_one_hot': False,
                                                  'encode_target_to_int': False, 'encode_target_to_one_hot': False}}
    for item in args_dict:
        if item not in ('model_dir', 'train_files'):
            param_dict['algo'][item] = args_dict[item]

    t = Trainer(data_path=args.train_files, model_path=args.model_dir, algo=DummyRegressor,
                params=param_dict)
    t.run()


if __name__ == '__main__':
    main()
