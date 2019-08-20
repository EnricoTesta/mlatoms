from sklearn.linear_model import LogisticRegression
from trainer import Trainer
import argparse


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='Logistic Regression Example')
    parser.add_argument(
        '--C',
        type=float,
        default=1,
        metavar='C',
        help='Inverse of regularization strength (default: 1)')
    parser.add_argument(
        '--penalty',
        type=str,
        default='l2',
        metavar='penalty',
        help='Used to specify the norm used in the penalization')
    parser.add_argument(
        '--model-dir',
        default='/mlatoms/test/modeldir',
        metavar='model_dir',
        help='The directory to store the model')
    parser.add_argument(
        '--train-files',
        default='/mlatoms/data/classification/binary/',
        metavar='train_files',
        help='The directory to fetch train data')

    args = parser.parse_args()
    return args


def main():
    # Training settings
    args = get_args()
    args_dict = vars(args)

    # Make param dict
    param_dict = {'algo': {}, 'fit': {}}
    for item in args_dict:
        if item not in ('model_dir', 'train_files'):
            param_dict['algo'][item] = args_dict[item]

    t = Trainer(train_data_path=args.train_files, model_path=args.model_dir, algo=LogisticRegression,
                params=param_dict, hypertune_loss='accuracy')
    t.run()


if __name__ == '__main__':
    main()
