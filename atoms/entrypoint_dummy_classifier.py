from trainer import Trainer
# from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
# from pandas import Series
# from numpy import ones
import argparse


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='Dummy Classifier')
    parser.add_argument(
        '--strategy',
        type=str,
        default='stratified',
        metavar='strategy',
        help="Used to specify a strategy. Can take values: 'prior', 'stratified',"
             " 'most_frequent', 'uniform', 'constant'")
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
        default='/mlatoms/data/classification/multi/',
        metavar='train_files',
        help='The directory to fetch train data')

    args = parser.parse_args()
    return args


# class DummyClassifier(BaseEstimator):
#
#     def __init__(self, strategy='most_frequent', constant=None):
#         self.strategy = strategy
#         self.constant = constant
#
#         self.output_shape = None
#         self.prediction_label = None
#         self.prediction_proba = None
#
#     def predict(self, x):
#         return self.prediction_label*ones((x.shape[0], ))
#
#     def predict_proba(self, x):
#         return self.prediction_proba*ones((x.shape[0], self.output_shape))
#
#     def fit(self, x, y):
#         counts = Series(y).value_counts(normalize=True)
#         self.prediction_label = counts.idxmax()
#         self.output_shape = counts.shape[0]
#         self.prediction_proba = 1/self.output_shape


def main():
    # Training settings
    args = get_args()
    args_dict = vars(args)

    # Make param dict
    param_dict = {'algo': {}, 'fit': {}}
    for item in args_dict:
        if item not in ('model_dir', 'train_files'):
            param_dict['algo'][item] = args_dict[item]

    t = Trainer(train_data_path=args.train_files, model_path=args.model_dir, algo=DummyClassifier,
                params=param_dict)
    t.run()


if __name__ == '__main__':
    main()
