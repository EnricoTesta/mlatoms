from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import make_scorer, roc_auc
from sklearn.model_selection import PredefinedSplit, KFold
from sklearn.datasets import load_breast_cancer
from pandas import read_csv
from trainer import Trainer
import numpy as np
import subprocess
import argparse
import psutil
import os


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='Light Gradient Boosting Classifier')
    parser.add_argument(
        '--time_left_for_this_task',
        type=int,
        default=60,
        metavar='time_left_for_this_task',
        help='Total time')
    parser.add_argument(
        '--per_run_time_limit',
        type=int,
        default=60,
        metavar='per_run_time_limit',
        help='Maximum tree depth for base learners, <=0 means no limit.')
    parser.add_argument(
        '--initial_configurations_via_metalearning',
        type=int,
        default=0,
        metavar='initial_configurations_via_metalearning',
        help='Maximum tree leaves for base learners')
    parser.add_argument(
        '--ensemble_size',
        type=int,
        default=100,
        metavar='ensemble_size',
        help='')
    parser.add_argument(
        '--ensemble_nbest',
        type=int,
        default=10,
        metavar='ensemble_nbest',
        help='')
    parser.add_argument(
        '--ml_memory_limit',
        type=int,
        default=0,
        metavar='ml_memory_limit',
        help='')
    parser.add_argument(
        '--resampling_strategy',
        type=str,
        default="PredefinedSplit",
        metavar='resampling_strategy',
        help='')
    parser.add_argument(
        '--resampling_strategy_arguments',
        type=str,
        default=None,
        metavar='resampling_strategy_arguments',
        help='')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        metavar='seed',
        help='')
    parser.add_argument(
        '--model-dir',
        type=str,
        default='/mlatoms/test/modeldir',
        metavar='model_dir',
        help='The directory to store the model')
    parser.add_argument(
        '--train-files',
        type=str,
        default='/mlatoms/data/classification/binary/',
        metavar='train_files',
        help='The directory to fetch train data')
    args = parser.parse_args()
    return args


class ASKLClassifierTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, x, y):
        estimator = self.algo(**self.params['algo'])
        if not bool(self.params['fit']):
            metric = make_scorer('roc_auc', roc_auc, greater_is_better=True, needs_proba=True)
        else:
            metric = self.params['fit']['metric']
        trained_estimator = estimator.fit(x, y, metric=metric)
        return trained_estimator.refit(x, y)

    def get_out_of_samples_prediction(self, x, y):
        return None

    def generalization_assessment(self, x, y):
        return None  # TODO: implement explicit cross validation procedure


def main():
    # Training settings
    args = get_args()
    args_dict = vars(args)

    # Interpret some of the input parameters

    # Available memory
    mem = psutil.virtual_memory()._asdict()
    args_dict['ml_memory_limit'] = int((mem['available'] / (pow(10, 6))) * 0.7)

    # CV strategy
    if args_dict['resampling_strategy'] == 'PredefinedSplit':
        args_dict['resampling_strategy'] = PredefinedSplit
    else:
        raise NotImplementedError

    try:
        local_file_path = os.path.join("/tmp/", "train_data.csv")
        subprocess.check_call(['gsutil', 'cp', args.train_files, local_file_path])
        data = read_csv(local_file_path)  # by convention the first column is always the target
    except FileNotFoundError:
        data = read_csv(args.train_files)  # for debug

    skf = KFold(n_splits=3, shuffle=True)
    folds = skf.split(data)
    pre_splits = np.zeros(data.shape[0])
    test_fold_counter = 0
    for train_idx, test_idx in folds:
        pre_splits[test_idx] = test_fold_counter
        test_fold_counter += 1
    args_dict['resampling_strategy_arguments'] = {'test_fold': pre_splits}

    # Make param dict
    param_dict = {'algo': {}, 'fit': {}}
    for item in args_dict:
        if item not in ('model_dir', 'train_files'):
            param_dict['algo'][item] = args_dict[item]

    t = ASKLClassifierTrainer(train_data_path=args.train_files, model_path=args.model_dir,
                              algo=AutoSklearnClassifier, params=param_dict, hypertune_loss='accuracy')
    t.run()


if __name__ == '__main__':
    main()
