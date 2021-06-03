import numpy as np
import hypertune
from shutil import rmtree
from pandas import DataFrame, Series, concat, unique
from validation import KFoldValidationSchema, StratifiedKFoldValidationSchema, GroupStrataKFoldValidationSchema
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler
from atoms import Atom, squeeze_proba
from scipy.stats import spearmanr, pearsonr


CLASSIFICATION_ESTIMATORS = ['LogisticRegression', 'LGBMClassifier',
                             'XGBClassifier', 'AutoSklearnClassifier',
                             'LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis',
                             'DummyClassifier', 'Sequential']
REGRESSION_ESTIMATORS = ['LinearRegression', 'LGBMRegressor']
BENCHMARK_ESTIMATORS = ['DummyClassifier']
CV_FOLDS = 3
DEFAULT_IMBALANCE_TOLERANCE = 0.20
HYPERTUNE_LOSSES = {'binary_crossentropy': 'log_loss',
                    'categorical_crossentropy': 'log_loss',
                    'accuracy': 'accuracy'}


def spearman_corrcoef(y, y_pred):
    try:
        cols = y_pred.shape[1]
    except IndexError:
        cols = 1
    if cols == 1:
        col_list = [1]
    else:
        col_list = [i for i in range(cols)]
    try:
        y_pred_synthetic = np.matmul(y_pred, col_list).reshape(-1, 1) / cols
    except ValueError:
        y_pred_synthetic = np.matmul(y_pred.reshape(-1, 1), col_list).reshape(-1, 1) / cols
    r, _ = spearmanr(y, y_pred_synthetic)
    return r


def pearson_corrcoef(y, y_pred):
    try:
        cols = y_pred.shape[1]
    except IndexError:
        cols = 1
    if cols == 1:
        col_list = [1]
    else:
        col_list = [i for i in range(cols)]
    try:
        y_pred_synthetic = np.matmul(y_pred, col_list).reshape(-1, 1) / cols
    except ValueError:
        y_pred_synthetic = np.matmul(y_pred.reshape(-1, 1), col_list).reshape(-1, 1) / cols
    r, _ = pearsonr(y.ravel(), y_pred_synthetic.ravel())
    return r


def get_imbalance_tolerance(n_samples):
    if n_samples < 1000:
        return 0.01
    elif n_samples < 100000:
        return 0.05
    else:
        return DEFAULT_IMBALANCE_TOLERANCE


class Trainer(Atom):
    """How do I know what trials I'm in??? It would be of great help in using HyperTune and setting model name.
    --> Can be found in TrainingOutput..."""

    def __init__(self, data_path=None, model_path=None, algo=None, params=None, hypertune_loss=None):
        super().__init__(data_path, model_path, algo, params)
        self.hypertune_loss = hypertune_loss
        self.problem_specs = None

    def _get_metrics_dict(self, target):
        metrics_dict = {}
        if self.algo.__name__ in CLASSIFICATION_ESTIMATORS:

            metrics_dict['accuracy'] = {'func': metrics.accuracy_score, 'kwargs': {}}
            metrics_dict['log_loss'] = {'func': metrics.log_loss, 'kwargs': {}}
            metrics_dict['matthews_corr'] = {'func': metrics.matthews_corrcoef, 'kwargs': {}}
            metrics_dict['spearman_corr'] = {'func': spearman_corrcoef, 'kwargs': {}}
            metrics_dict['pearson_corr'] = {'func': pearson_corrcoef, 'kwargs': {}}
            if self.problem_specs['binary']:
                metrics_dict['roc_auc'] = {'func': metrics.roc_auc_score, 'kwargs': {}}  # average: 'macro' is default
                metrics_dict['hinge_loss'] = {'func': metrics.hinge_loss, 'kwargs': {}}
                metrics_dict['f1'] = {'func': metrics.f1_score, 'kwargs': {}}
                metrics_dict['precision'] = {'func': metrics.precision_score, 'kwargs': {}}
                metrics_dict['recall'] = {'func': metrics.recall_score, 'kwargs': {}}
                metrics_dict['fbeta'] = {'func': metrics.fbeta_score, 'kwargs': {'average': 'binary'}}
            else:  # multi-class
                metrics_dict['f1'] = {'func': metrics.f1_score, 'kwargs': {'average': 'weighted'}}
                metrics_dict['precision'] = {'func': metrics.precision_score, 'kwargs': {'average': 'weighted'}}
                metrics_dict['recall'] = {'func': metrics.recall_score, 'kwargs': {'average': 'weighted'}}
                metrics_dict['fbeta'] = {'func': metrics.fbeta_score, 'kwargs': {'average': 'weighted'}}

        elif self.algo.__name__ in REGRESSION_ESTIMATORS:
            metrics_dict['MSE'] = {'func': metrics.mean_squared_error, 'kwargs': {}}
            metrics_dict['MAE'] = {'func': metrics.mean_absolute_error, 'kwargs': {}}
            # metrics_dict['MAPE'] = {'func': metrics.mean_absolute_percentage_error, 'kwargs': {}}
        else:
            raise NotImplementedError
        return metrics_dict

    def _get_score_method(self):
        if self.algo.__name__ in CLASSIFICATION_ESTIMATORS:
            return 'predict_proba'
        elif self.algo.__name__ in REGRESSION_ESTIMATORS:
            return 'predict'
        else:
            raise NotImplementedError

    def _get_estimator_type(self):
        if self.algo.__name__ in CLASSIFICATION_ESTIMATORS:
            return 'classification'
        elif self.algo.__name__ in REGRESSION_ESTIMATORS:
            return 'regression'
        else:
            raise NotImplementedError

    @staticmethod
    def infer_problem_specs(x, y):
        """Returns a dictionary with 'type': regression/classification, 'balanced': True/False/None,
         'binary': True/False/None, 'n_dimensions': integer, 'n_samples': integer"""
        d = {'n_dimensions': x.shape[1], 'n_samples': x.shape[0]}
        if isinstance(y, DataFrame):
            y = y.dot(range(y.shape[1]))
        yu, yu_counts = np.unique(y, return_counts=True)
        if len(yu) >= 100 or np.count_nonzero(yu - np.around(yu)) > 0:
            d['type'] = 'regression'
            d['binary'] = None
            d['balanced'] = None
        else:
            d['type'] = 'classification'
        if d['type'] == 'classification':
            d['binary'] = len(yu) == 2
            response_values = y.value_counts(normalize=True)
            d['balanced'] = bool((response_values.max() - response_values.min())/2 <= get_imbalance_tolerance(d['n_samples']))
        return d

    @staticmethod
    def _build_stratification_variable(stratification_df):
        # Build stratification variable
        tmp_var = stratification_df.astype(str)
        stratification_variable_name = 'stratified_'
        strat_var = None
        for column in tmp_var.columns:
            try:
                strat_var += tmp_var[column]
            except TypeError:
                strat_var = tmp_var[column]
            stratification_variable_name += column + '_'
        return DataFrame(data=strat_var.values, columns=[stratification_variable_name[0:-1]], index=strat_var.index), \
               unique(strat_var)

    def generate_validation_schemas(self, stratification):
        # TODO: move this to validation.py as a function
        schemas = {}
        if stratification is None:
            schemas['KFold'] = {'schema': KFoldValidationSchema(params={'n_splits': CV_FOLDS, 'shuffle': False}),
                                'kwargs': {}}
        else:
            strat_var, _ = self._build_stratification_variable(stratification)
            schemas['GroupStrataKFold'] = {'schema': GroupStrataKFoldValidationSchema(params={'n_splits': CV_FOLDS,
                                                                                            'shuffle': False}),
                                          'kwargs': {'stratification_variable': strat_var}}
            schemas['StratifiedKFold'] = {'schema': StratifiedKFoldValidationSchema(params={'n_splits': CV_FOLDS,
                                                                                            'shuffle': False}),
                                          'kwargs': {'stratification_variable': strat_var}}
        for key, value in schemas.items():
            value['kwargs']['balanced'] = self.problem_specs['balanced']
        return schemas

    @staticmethod
    def get_fit_params_dict(x, y, train_idx, validation_idx):
        return {}  # implemented only in subclasses if needed

    @staticmethod
    def get_train_information(trained_model):
        return {}  # implemented only in subclasses if needed

    def validation_assessment(self, x, y, validation_schemas):
        """
        :param x: Features DataFrame
        :param y: Response Series
        :param validation_schemas: dict of dicts (ValidationSchema objects + dict of kwargs)
        :return: validation, stratified_validation, pred
        """
        validation_df_list = []
        stratified_validation_df_list = []
        pred_df_list = []
        train_information_df_list = []
        metrics_dict = self._get_metrics_dict(y)
        for key, item in validation_schemas.items():

            # Evaluate validation schema
            for train_idx, validation_idx in item['schema'].split(x, y, **item['kwargs']):  # sklearn-like generators
                trained_model = self.fit(x.iloc[train_idx, :], y.iloc[train_idx], self.get_algo_params(validation=True),
                                         self.get_fit_params(**{**self.get_fit_params_dict(x, y, train_idx,
                                                                                           validation_idx),
                                                                **{'validation': True}}))
                validation, stratified_validation, pred = self.get_model_performance(trained_model,
                                                                                     x.iloc[validation_idx, :],
                                                                                     y.iloc[validation_idx],
                                                                                     metrics_dict)
                validation_df = DataFrame(validation, index=[0])
                validation_df['validation_schema'] = key
                validation_df_list.append(validation_df)

                stratified_validation_df = DataFrame(stratified_validation)
                stratified_validation_df['validation_schema'] = key
                stratified_validation_df_list.append(stratified_validation_df)

                pred['validation_schema'] = key
                pred_df_list.append(pred)

                train_information_df = DataFrame(self.get_train_information(trained_model), index=[0])
                train_information_df['validation_schema'] = key
                train_information_df_list.append(train_information_df)

        return concat(validation_df_list, axis=0),\
               concat(stratified_validation_df_list, axis=0),\
               concat(pred_df_list, axis=0),\
               concat(train_information_df_list, axis=0)

    def _compute_metrics(self, labeled_predictions, metrics_dict):
        d = {}
        for name, metric in metrics_dict.items():
            try:
                if name == 'fbeta':
                    metric_score = metric['func'](labeled_predictions['ground_truth'].values,
                                                  labeled_predictions['pred_response'].values, 1, **metric['kwargs'])
                elif name == 'pearson_corr' or name == 'spearman_corr':
                    metric_score = metric['func'](labeled_predictions['ground_truth'].values,
                                          squeeze_proba(labeled_predictions.iloc[:, 0:-2]), **metric['kwargs'])  # y_true, y_pred
                else:
                    try:
                        metric_score = metric['func'](labeled_predictions['ground_truth'].values,
                                              labeled_predictions.iloc[:, 0:-2].values, **metric['kwargs'])  # y_true, y_pred
                    except ValueError:  # metric does not support probabilities
                        metric_score = metric['func'](labeled_predictions['ground_truth'].values,
                                                      labeled_predictions['pred_response'].values, **metric['kwargs'])
            except:
                metric_score = np.nan
            d['test_' + name] = metric_score
        d['benchmark'] = int(self.algo.__name__ in BENCHMARK_ESTIMATORS)
        d['algo'] = self.algo.__name__
        return d

    def _compute_feature_exposure(self, labeled_predictions, features):
        pearson = []
        spearman = []
        if labeled_predictions.shape[1] == 2: # no probabilities involved
            pred = labeled_predictions['pred_response'].values
        else:
            pred = squeeze_proba(labeled_predictions.iloc[:, 0:-2])
        for f in features:
            relevant_indexes = features[f].notnull()
            p = pearson_corrcoef(features[f][relevant_indexes], pred[relevant_indexes])
            s = spearman_corrcoef(features[f][relevant_indexes], pred[relevant_indexes])
            if not np.isnan(p):
                pearson.append(p)
            if not np.isnan(s):
                spearman.append(s)
        return np.std(pearson, ddof=1), np.std(spearman, ddof=1)


    def get_model_performance(self, trained_model, x, y, metrics_dict):

        # Get out-of-fold predictions
        pred = DataFrame(self.predict(trained_model, x), index=x.index)
        predictions_col_names = []
        if pred.shape[1] == 1:
            predictions_col_names.append("pred_response")  # either class labels or regression values
        else:
            for i in range(pred.shape[1]):
                predictions_col_names.append("probability_" + str(i))
            pred['pred_response'] = pred.idxmax(axis=1)  # add class label
            predictions_col_names.append('pred_response')
        pred.columns = predictions_col_names

        # Get metrics
        if y.shape[1] == 1:
            labeled_pred = pred.merge(y.rename(columns={y.columns[0]: 'ground_truth'}),
                                      left_index=True, right_index=True, copy=False)
        else:
            # condense 1-hot response
            labeled_pred = pred.merge(Series(y.dot(range(y.shape[1])), name='ground_truth'),
                                      left_index=True, right_index=True, copy=False)
        validation = self._compute_metrics(labeled_predictions=labeled_pred, metrics_dict=metrics_dict)
        validation['test_feature_exposure_pearson'], \
        validation['test_feature_exposure_spearman'] = self._compute_feature_exposure(labeled_pred, x)

        # Get stratified metrics
        # TODO: add stratified feature exposure
        stratified_validation = {'strata': [], 'strata_weight': [], 'benchmark': [], 'algo': []}
        for metric_name in metrics_dict:
            stratified_validation['test_' + metric_name] = []
        if "STRATIFICATION_COLUMN" in self.info:
            strat_df, strat_values = self._build_stratification_variable(self.data[self.info["STRATIFICATION_COLUMN"]])
            strat_pred = strat_df.merge(pred, left_index=True, right_index=True)
            if y.shape[1] == 1:
                strat_pred = strat_pred.merge(y.rename(columns={y.columns[0]: 'ground_truth'}),
                                                       left_index=True, right_index=True, copy=False)
            else:
                strat_pred = strat_pred.merge(Series(y.dot(range(y.shape[1])), name='ground_truth'),
                                              left_index=True, right_index=True, copy=False)
            for value in strat_values:
                relevant_data = strat_pred.loc[strat_pred[strat_pred.columns[0]] == value].drop(columns=strat_df.columns[0])  # get labeled_predictions-like DataFrame
                if not relevant_data.empty:  # validation schema may not always contain all stratification values
                    stratified_validation['strata'].append(value)
                    stratified_validation['strata_weight'].append(relevant_data.shape[0] / strat_pred.shape[0])
                    d = self._compute_metrics(labeled_predictions=relevant_data, metrics_dict=metrics_dict)
                    for k, v in d.items():
                        stratified_validation[k].append(v)

        # Reset prediction index
        pred.reset_index(inplace=True)

        return validation, stratified_validation, pred

    def run(self):

        # Fetch data
        self.retrieve_data()
        self.retrieve_metadata()
        self.read_info()
        self.read_metadata()
        self.read_data(**self.params['read'])

        y = self.data[[col for col in self.data.columns if self.info["TARGET_COLUMN"] in col]]
        if self.algo.__name__ in CLASSIFICATION_ESTIMATORS:
            if isinstance(y, DataFrame):
                flag_list = []
                for col in y:
                    flag_list.append(y[col].apply(lambda x: x - int(x) != 0).any())  # THIS IS SLOW
                non_integer_encoding_flag = max(flag_list)
            elif isinstance(y, Series):
                non_integer_encoding_flag = y.apply(lambda x: x - int(x) != 0).any()  # THIS IS SLOW
            else:
                raise TypeError("Target should be either DataFrame or Series, got %s." % type(y))
            if non_integer_encoding_flag:
                raise ValueError("Target variable for classification algorithms must be integer encoded.")

        try:
            special_column_list = [col for col in self.data.columns if self.info["TARGET_COLUMN"] in col] + self.info["STRATIFICATION_COLUMN"]
        except KeyError:
            special_column_list = [col for col in self.data.columns if self.info["TARGET_COLUMN"] in col]
        x = self.data.iloc[:, np.isin(self.data.columns, special_column_list, invert=True)]
        self.problem_specs = self.infer_problem_specs(x, y)

        # Generalization Assessment
        force_stratify_on_target = False
        if self.problem_specs["type"] == "classification" and not self.problem_specs["balanced"]:
            force_stratify_on_target = True  # ensure you have all classes in each fold

        stratification_column_list = []
        if "STRATIFICATION_COLUMN" in self.info:
            stratification_column_list += self.info["STRATIFICATION_COLUMN"]
        if force_stratify_on_target:
            stratification_column_list += [col for col in self.data.columns if self.info["TARGET_COLUMN"] in col]
        stratification_column_list = list(set(stratification_column_list))  # remove duplicates

        if stratification_column_list:
            strat_df = self.data[stratification_column_list]
        else:
            strat_df = None

        schemas = self.generate_validation_schemas(strat_df)
        self.validation, self.stratified_validation, self.predictions, train_info = self.validation_assessment(x, y, schemas)

        # Fit model on entire train data
        if self.problem_specs["type"] == "classification" and not self.problem_specs["balanced"]:
            rus = RandomUnderSampler(sampling_strategy='not minority', replacement=False)
            resampled_index, _ = rus.fit_resample(x.index.values.reshape(-1, 1), y)
            resampled_index = resampled_index.flatten()
            self.trained_model = self.fit(x.loc[resampled_index, :],
                                          y.loc[resampled_index],
                                          self.get_algo_params(validation=False, train_info=train_info),
                                          self.get_fit_params(validation=False))
        else:
            self.trained_model = self.fit(x, y, self.get_algo_params(validation=False, train_info=train_info),
                                          self.get_fit_params(validation=False))

        # Export
        unique_id = self.generate_unique_id()
        self.export_model_file(self.trained_model, 'model_' + unique_id)
        export_params_dict = {'algo': {**self.get_algo_params(validation=False, train_info=train_info)},
                              'fit': {**self.get_fit_params(validation=False)}}
        if 'custom_loss' in export_params_dict['algo'].keys(): # object is not JSON serializable
            export_params_dict['algo']['custom_loss'] = export_params_dict['algo']['custom_loss'].__class__.__name__
        self.export_file(export_params_dict, 'params_' + unique_id + '.json')
        self.export_file(self.predictions, 'predictions_' + unique_id + '.csv')
        self.export_file(self.validation, 'info_' + unique_id + '.csv')
        self.export_file(self.stratified_validation, 'stratified_info_' + unique_id + '.csv')

        # Clean-up
        rmtree(self.local_path)

        # Report Loss to Hypertune
        if self.hypertune_loss is None or self.validation is None:
            return
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=self.hypertune_loss,
            metric_value=self.validation['test_' + HYPERTUNE_LOSSES[self.hypertune_loss]].mean())

    def export_model_file(self, obj, file_name):
        super().export_file(obj, file_name + '.pkl')
