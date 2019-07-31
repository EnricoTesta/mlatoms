from lightgbm import LGBMClassifier
from trainer import Trainer
import argparse


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='Light Gradient Boosting Classifier')
    parser.add_argument(
        '--boosting_type',
        type=str,
        default='gbdt',
        metavar='boosting_type',
        help='Can be gbdt, dart, goss, rf')
    parser.add_argument(
        '--max_depth',
        type=int,
        default=-1,
        metavar='max_depth',
        help='Maximum tree depth for base learners, <=0 means no limit.')
    parser.add_argument(
        '--num_leaves',
        type=int,
        default=31,
        metavar='num_leaves',
        help='Maximum tree leaves for base learners')
    parser.add_argument(
        '--n_estimators',
        type=int,
        default=100,
        metavar='n_estimators',
        help='Number of boosted trees to fit.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.1,
        metavar='learning_rate',
        help='Boosting learning rate.')
    parser.add_argument(
        '--subsample',
        type=float,
        default=1,
        metavar='subsample',
        help='Subsample ratio of the training instance')
    parser.add_argument(
        '--colsample_bytree',
        type=float,
        default=1,
        metavar='colsample_bytree',
        help='Subsample ratio of columns when constructing each tree')
    parser.add_argument(
        '--reg_alpha',
        type=float,
        default=0,
        metavar='reg_alpha',
        help='L1 regularization term on weights')
    parser.add_argument(
        '--reg_lambda',
        type=float,
        default=0,
        metavar='reg_lambda',
        help='L2 regularization term on weights')
    parser.add_argument(
        '--importance_type',
        type=str,
        default='split',
        metavar='importance_type',
        help='Can be split or gain')
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        metavar='model_dir',
        help='The directory to store the model')
    parser.add_argument(
        '--train-files',
        type=str,
        default=None,
        metavar='train_files',
        help='The directory to fetch train data')
    parser.add_argument(
        '--debug',
        default=0,
        type=float,
        metavar='debug',
        help='Debug flag')
    args = parser.parse_args()
    return args


def main():
    # Training settings
    args = get_args()
    args_dict = vars(args)

    # Make param dict
    param_dict = {}
    for item in args_dict:
        if item not in ('model_dir', 'train_files', 'debug'):
            param_dict[item] = args_dict[item]

    t = Trainer(train_data_path=args.train_files, model_path=args.model_dir, algo=LGBMClassifier,
                params=param_dict, hypertune_loss='accuracy', debug=args.debug)
    t.run()


if __name__ == '__main__':
    main()
