from lightgbm import LGBMRegressor, train, Dataset
from trainer import Trainer
from loss import *
import argparse


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='Light Gradient Boosting')
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
        default=20,
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
        default= '/tmp/pycharm_project_661/test/modeldir', # '/mlatoms/test/modeldir',
        metavar='model_dir',
        help='The directory to store the model')
    parser.add_argument(
        '--train-files',
        type=str,
        default= '~/data/', # '/gauth/NTEST/', # '/mlatoms/data/classification/multi/numer/train/',
        metavar='train_files',
        help='The directory to fetch train data')
    parser.add_argument(
        '--hypertune-loss',
        default=None,
        metavar='hypertune_loss',
        help='Hypertune loss name')
    parser.add_argument(
        '--custom-loss',
        default='CustomMSEVarianceDecay',
        metavar='custom_loss',
        help='Custom loss name')

    args = parser.parse_args()
    return args


class LGBMTrainer(Trainer):

    @staticmethod
    def get_fit_params_dict(x, y, train_idx, validation_idx):
        return {'validation_data': [(x.iloc[validation_idx, :], y.iloc[validation_idx])]}

    @staticmethod
    def get_train_information(trained_model):
        try:
            if trained_model.best_iteration_ is None or trained_model.best_iteration_ == 0:
                return {'n_estimators': trained_model.n_estimators}
            else:
                return {'n_estimators': trained_model.best_iteration_}
        except AttributeError:
            # using custom loss
            # Attribute 'best_iteration_' --> 'best_iteration'
            # best_iteration set to 0 instead of None
            if trained_model.best_iteration is None or trained_model.best_iteration == 0:
                return {'n_estimators': trained_model.params['num_iterations']}
            else:
                return {'n_estimators': trained_model.best_iteration}

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

    def _get_train_dict(self, algo_params, fit_params):
        kwargs = {}
        params = {}
        algo_params_copy = algo_params.copy()
        fit_params_copy = fit_params.copy()

        # Iterate on algo_params
        for key in list(algo_params_copy.keys()):
            if key == 'n_estimators':
                kwargs['num_boost_round'] = algo_params_copy.pop('n_estimators')
            elif key == 'custom_loss':
                kwargs['fobj'] = algo_params_copy['custom_loss'].objective
                kwargs['feval'] = algo_params_copy['custom_loss'].eval
            else:
                params[key] = algo_params_copy[key]

        # Iterate on fit_params
        for key in list(fit_params_copy.keys()):
            if key == 'early_stopping_rounds':
                kwargs['early_stopping_rounds'] = fit_params_copy.pop('early_stopping_rounds')
            elif key == 'eval_set':
                tmp = fit_params_copy.pop('eval_set')
                kwargs['valid_sets'] = [Dataset(tmp[0][0], tmp[0][1], init_score=None)]
            else:
                params[key] = fit_params_copy[key]

        return kwargs, params

    def fit(self, x, y, algo_params, fit_params):
        # Redefine fit method to use lgbm Python API
        if 'custom_loss' in algo_params.keys():
            train_data = Dataset(x, label=y, init_score=None)
            kwargs, params = self._get_train_dict(algo_params, fit_params)
            return train(params, train_data, **kwargs)
        return super().fit(x, y, algo_params, fit_params)


def main():
    # Training settings
    args = get_args()
    args_dict = vars(args)

    # Make param dict
    param_dict = {'algo': {}, 'fit': {}, 'read': {'encode_features_to_int': False, 'encode_features_to_one_hot': True,
                                                  'encode_target_to_int': False, 'encode_target_to_one_hot': False}}
    for item in args_dict:
        if item not in ('model_dir', 'train_files', 'hypertune_loss', 'early_stopping_rounds', 'custom_loss'):
            param_dict['algo'][item] = args_dict[item]
        elif item == 'early_stopping_rounds' and args_dict[item] is not None:
            param_dict['fit'][item] = args_dict[item]  # must specify validation data and evaluation metric @ fit time
        elif item == 'custom_loss':
            param_dict['algo']['custom_loss'] = eval(args_dict[item] + '()') # instantiate custom loss

    t = LGBMTrainer(data_path=args.train_files, model_path=args.model_dir, algo=LGBMRegressor,
                    params=param_dict, hypertune_loss=args.hypertune_loss)
    t.run()


if __name__ == '__main__':
    main()
