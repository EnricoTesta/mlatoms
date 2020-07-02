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
        default=2000,
        metavar='n_estimators',
        help='Number of boosted trees to fit.')
    parser.add_argument(
        '--early_stopping_rounds',
        type=int,
        default=None,
        metavar='early_stopping_rounds',
        help='Maximum number of iterations allowed without metric improvement on validation data')
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


class LGBMTrainer(Trainer):

    @staticmethod
    def get_fit_params_dict(x, y, train_idx, validation_idx):
        return {'validation_data': [(x.iloc[validation_idx, :], y.iloc[validation_idx])]}

    @staticmethod
    def get_train_information(trained_model):
        if trained_model._best_iteration is None:
            return {'n_estimators': trained_model.n_estimators}
        else:
            return {'n_estimators': trained_model._best_iteration}

    def get_algo_params(self, **kwargs):
        params = self.params['algo'].copy()
        if kwargs['validation']:
            pass
        else:
            # Replace with train_info whenever possible
            train_info = kwargs['train_info'].max().to_dict()
            for key, value in params.items():
                if key in train_info.keys():
                    params[key] = train_info[key]
        return params

    def get_fit_params(self, **kwargs):
        # Account for early stopping
        params = self.params['fit'].copy()
        if kwargs['validation']:
            if 'early_stopping_rounds' in params and params['early_stopping_rounds'] is not None:
                params['eval_set'] = kwargs['validation_data']  # evaluated on the metric set by the model
        else:
            try:
                params.pop('early_stopping_rounds')  # not needed in final training
            except KeyError:
                pass
        return params


def main():
    # Training settings
    args = get_args()
    args_dict = vars(args)

    # Make param dict
    param_dict = {'algo': {}, 'fit': {}, 'read': {'encode_features_to_int': False, 'encode_features_to_one_hot': True,
                                                  'encode_target_to_int': True, 'encode_target_to_one_hot': False}}
    for item in args_dict:
        if item not in ('model_dir', 'train_files', 'hypertune_loss', 'early_stopping_rounds'):
            param_dict['algo'][item] = args_dict[item]
        elif item == 'early_stopping_rounds' and args_dict[item] is not None:
            param_dict['fit'][item] = args_dict[item]  # must specify validation data and evaluation metric @ fit time

    # Enable GPU
    param_dict['algo']['device'] = 'gpu'

    t = LGBMTrainer(data_path=args.train_files, model_path=args.model_dir, algo=LGBMClassifier,
                    params=param_dict, hypertune_loss=args.hypertune_loss)
    t.run()


if __name__ == '__main__':
    main()
