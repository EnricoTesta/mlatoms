from xgboost import XGBClassifier
from trainer import Trainer
import argparse


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='Extreme Gradient Boosting Classifier')
    parser.add_argument(
        '--booster',
        type=str,
        default='gbtree',
        metavar='booster',
        help='Can be gbtree, gblinear, or dart.')
    parser.add_argument(
        '--max_depth',
        type=int,
        default=3,
        metavar='max_depth',
        help='Maximum tree depth for base learners, 0 means no limit.')
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=-1,
        metavar='n_jobs',
        help='Number of threads to use. Negative means max possible.')
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
        help='Can be gain, weight, cover, total_gain, total_cover')
    parser.add_argument(
        '--model-dir',
        type=str,
        default='/mlatoms/test/modeldir',
        metavar='model_dir',
        help='The directory to store the model')
    parser.add_argument(
        '--train-files',
        type=str,
        default='/mlatoms/data/classification/binary/missing/train/',
        metavar='train_files',
        help='The directory to fetch train data')
    parser.add_argument(
        '--hypertune-loss',
        default=None,
        metavar='hypertune_loss',
        help='Hypertune loss name')
    args = parser.parse_args()
    return args


class XGBClassifierTrainer(Trainer):

    @staticmethod
    def predict(estimator, x):
        try:
            return estimator.predict_proba(x, validate_features=False)
        except:
            return estimator.predict(x, validate_features=False)


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

    t = XGBClassifierTrainer(data_path=args.train_files, model_path=args.model_dir, algo=XGBClassifier,
                             params=param_dict, hypertune_loss=args.hypertune_loss)
    t.run()


if __name__ == '__main__':
    main()
