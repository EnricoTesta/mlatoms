from pandas import concat, Series
from auto_ml import Predictor
from trainer import Trainer
import argparse


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='Light Gradient Boosting Classifier')
    parser.add_argument(
        '--n_splits',
        type=int,
        default=3,
        metavar='n_splits',
        help='CV splits')
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


class AMLClassifierTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, x, y):
        estimator = self.algo(**self.params['algo'])
        df_train = concat([x, Series(y, name='target')], axis=1)
        return estimator.train(df_train, cv=self.params['fit']['cv'])

    def get_out_of_samples_prediction(self, x, y):
        return None

    def generalization_assessment(self, x, y):
        return None  # TODO: implement explicit cross validation procedure


def main():
    # Training settings
    args = get_args()
    args_dict = vars(args)

    # Make param dict
    param_dict = {'algo': {}, 'fit': {}}
    param_dict['algo']['type_of_estimator'] = 'classifier'
    param_dict['algo']['column_descriptions'] = {'target': 'output'}
    param_dict['fit']['cv'] = args_dict['n_splits']
    for item in args_dict:
        if item not in ('cv', 'model_dir', 'train_files', 'debug'):
            param_dict['algo'][item] = args_dict[item]

    t = AMLClassifierTrainer(train_data_path=args.train_files, model_path=args.model_dir,
                             algo=Predictor, params=param_dict, hypertune_loss='accuracy',
                             debug=args.debug)
    t.run()


if __name__ == '__main__':
    main()
