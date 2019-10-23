from category_encoders.one_hot import OneHotEncoder
from preprocessor import Encoder
import argparse


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='Ordinal Encoder')
    parser.add_argument(
        '--cols',
        default=['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'M1',
                 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12'],
        metavar='cols',
        nargs='*',
        help='List of columns to encode')
    parser.add_argument(
        '--handle_unknown',
        default='value',
        metavar='handle_unknown',
        help='How to encode unknown values')
    parser.add_argument(
        '--handle_missing',
        default='return_nan',
        metavar='handle_missing',
        help='How to encode missing values')
    parser.add_argument(
        '--use_cat_names',
        default=False,
        metavar='use_cat_names',
        help='Use catetegory names in column names')
    parser.add_argument(
        '--model-dir',
        default='/mlatoms/test/modeldir',
        metavar='model_dir',
        help='The directory to store the object')
    parser.add_argument(
        '--train-files',
        default='/mlatoms/data/classification/binary/heterogeneous/train/',
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

    t = Encoder(data_path=args.train_files, model_path=args.model_dir, algo=OneHotEncoder, params=param_dict)
    t.run()


if __name__ == '__main__':
    main()
