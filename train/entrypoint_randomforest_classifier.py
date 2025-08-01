from lightgbm import LGBMClassifier
from trainer import Trainer
import argparse


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='Light Gradient Boosting Random Forest Classifier')
    parser.add_argument(
        '--boosting_type',
        type=str,
        default='rf',
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
        default=2000,
        metavar='n_estimators',
        help='Number of boosted trees to fit.')
    parser.add_argument(
        '--bagging_freq',
        type=int,
        default=1,
        metavar='bagging_freq',
        help='Perform bagging every k iteration')
    parser.add_argument(
        '--feature_fraction',
        type=float,
        default=0.1,
        metavar='feature_fraction',
        help='Subsample ratio of columns when constructing each tree. 0 < feature_fraction < 1.')
    parser.add_argument(
        '--bagging_fraction',
        type=float,
        default=0.8,
        metavar='bagging_fraction',
        help='Subsample ratio of samples when constructing each tree. 0 < bagging_fraction < 1.')
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
        default='/mlatoms/test/modeldir',
        metavar='model_dir',
        help='The directory to store the model')
    parser.add_argument(
        '--train-files',
        type=str,
        default='/mlatoms/data/classification/binary/ieee-fraud/train/',
        metavar='train_files',
        help='The directory to fetch train data')
    parser.add_argument(
        '--hypertune-loss',
        default=None,
        metavar='hypertune_loss',
        help='Hypertune loss name')

    args = parser.parse_args()
    return args


def main():
    # Training settings
    args = get_args()
    args_dict = vars(args)

    # Make param dict
    param_dict = {'algo': {}, 'fit': {}, 'read': {'encode_features_to_int': False, 'encode_features_to_one_hot': True,
                                                  'encode_target_to_int': True, 'encode_target_to_one_hot': False}}
    for item in args_dict:
        if item not in ('model_dir', 'train_files', 'hypertune_loss'):
            param_dict['algo'][item] = args_dict[item]

    t = Trainer(data_path=args.train_files, model_path=args.model_dir, algo=LGBMClassifier,
                params=param_dict, hypertune_loss=args.hypertune_loss)
    t.run()


if __name__ == '__main__':
    main()
