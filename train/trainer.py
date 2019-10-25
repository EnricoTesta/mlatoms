import numpy as np
import hypertune
from shutil import rmtree
from pandas import DataFrame, concat
from sklearn.model_selection import StratifiedKFold, KFold, PredefinedSplit, cross_validate
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler
from atoms import Atom


CLASSIFICATION_ESTIMATORS = ['LogisticRegression', 'LGBMClassifier',
                             'XGBClassifier', 'AutoSklearnClassifier',
                             'LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis', 'DummyClassifier']
REGRESSION_ESTIMATORS = ['LinearRegression']
BENCHMARK_ESTIMATORS = ['DummyClassifier']
CV_FOLDS = 3
IMBALANCE_TOLERANCE = 0.01
HYPERTUNE_LOSSES = {'binary_crossentropy': 'log_loss',
                    'categorical_crossentropy': 'log_loss',
                    'accuracy': 'accuracy'}


class ImbalancedPredefinedSplit(PredefinedSplit):

    def __init__(self, test_fold=None):
        super().__init__(test_fold)

    def split(self, X=None, y=None, groups=None):
        rus = RandomUnderSampler(sampling_strategy='not minority', replacement=False)
        for original_train_index, original_test_index in super(ImbalancedPredefinedSplit, self).split(X, y, groups):
            resampled_train_index, _ = rus.fit_resample(X.index[original_train_index].values.reshape(-1, 1),
                                                        y.iloc[original_train_index])
            resampled_mask = [item in resampled_train_index for item in X.index]
            resampled_train_rows = [i for i, item in enumerate(resampled_mask) if item is True]
            yield np.asarray(resampled_train_rows), original_test_index


class Trainer(Atom):
    """How do I know what trials I'm in??? It would be of great help in using HyperTune and setting model name.
    --> Can be found in TrainingOutput..."""

    def __init__(self, data_path=None, model_path=None, algo=None, params=None, hypertune_loss=None):
        super().__init__(data_path, model_path, algo, params)
        self.hypertune_loss = hypertune_loss
        self.problem_specs = None

    def _get_scoring_list(self, target):
        scoring_dict = {}
        if self.algo.__name__ in CLASSIFICATION_ESTIMATORS:

            scoring_dict['accuracy'] = metrics.make_scorer(metrics.accuracy_score)
            scoring_dict['log_loss'] = metrics.make_scorer(metrics.log_loss, greater_is_better=False, needs_proba=True)
            scoring_dict['matthews_corr'] = metrics.make_scorer(metrics.matthews_corrcoef)
            if target.unique().shape[0] == 2:  # binary
                scoring_dict['roc_auc'] = metrics.make_scorer(metrics.roc_auc_score)  # average: 'macro' is default
                scoring_dict['hinge_loss'] = metrics.make_scorer(metrics.hinge_loss, greater_is_better=False)
                scoring_dict['f1'] = metrics.make_scorer(metrics.f1_score)
                scoring_dict['precision'] = metrics.make_scorer(metrics.precision_score)
                scoring_dict['recall'] = metrics.make_scorer(metrics.recall_score)
                scoring_dict['fbeta'] = metrics.make_scorer(metrics.fbeta_score, beta=1, average='binary')
            else:  # multi-class
                scoring_dict['f1'] = metrics.make_scorer(metrics.f1_score, average='macro')
                scoring_dict['precision'] = metrics.make_scorer(metrics.precision_score, average='macro')
                scoring_dict['recall'] = metrics.make_scorer(metrics.recall_score, average='macro')
                scoring_dict['fbeta'] = metrics.make_scorer(metrics.fbeta_score, beta=1, average='macro')

        elif self.algo.__name__ in REGRESSION_ESTIMATORS:
            raise NotImplementedError
        else:
            raise NotImplementedError
        return scoring_dict

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
            d['balanced'] = bool((response_values.max() - response_values.min())/2 <= IMBALANCE_TOLERANCE)
        return d

    def generate_folds(self, x, y, stratification=None):
        # TODO: manage shuffle without messing up with original DataFrame indexes
        test_folds = -np.ones(y.shape)
        counter = 0

        # Stratification
        if stratification is None:
            for _, test_index in KFold(n_splits=CV_FOLDS, shuffle=False).split(x):
                test_folds[test_index] = counter
                counter += 1
        else:
            # Build stratification variable
            tmp_var = stratification.astype(str)
            for column in tmp_var.columns:
                try:
                    strat_var += tmp_var[column]
                except NameError:
                    strat_var = tmp_var[column]
            # Draw folds
            for _, test_index in StratifiedKFold(n_splits=CV_FOLDS, shuffle=False).split(x, strat_var):
                test_folds[test_index] = counter
                counter += 1

        # Imbalance
        if self.problem_specs["type"] == "classification":
            if self.problem_specs["balanced"]:
                return PredefinedSplit(test_folds)
            else:
                return ImbalancedPredefinedSplit(test_folds)
        else:
            return PredefinedSplit(test_folds)

    def generalization_assessment(self, x, y, cv):
        validation = cross_validate(self.algo(**self.params['algo']), x, y=y, scoring=self._get_scoring_list(y),
                                    return_train_score=True, return_estimator=True, n_jobs=-1, cv=cv)
        estimators = validation.pop('estimator')

        # Get out-of-fold predictions
        pred = concat([DataFrame(self.predict(fit_model, x.loc[x.index[cv.test_fold == idx], :]),
                                 index=x.index[cv.test_fold == idx])
                      for idx, fit_model in enumerate(estimators)], axis=0)
        predictions_col_names = []
        if pred.shape[1] == 1:
            predictions_col_names.append("value")  # either class labels or regression values
        else:
            for i in range(pred.shape[1]):
                predictions_col_names.append("probability_" + str(i))
        pred.columns = predictions_col_names
        pred.reset_index(inplace=True)

        # Add info to validation
        validation['benchmark'] = int(self.algo.__name__ in BENCHMARK_ESTIMATORS)
        # Logloss correction
        validation['train_log_loss'] = -validation['train_log_loss']
        validation['test_log_loss'] = -validation['test_log_loss']
        # Algo name
        validation['algo'] = self.algo.__name__
        return validation, pred

    def run(self):

        # Fetch data
        self.retrieve_data()
        self.read_info()
        self.read_data()

        y = self.data[self.info["TARGET_COLUMN"]]
        if self.algo.__name__ in CLASSIFICATION_ESTIMATORS and y.apply(lambda x: x - int(x) != 0).any():  # THIS IS SLOW
            raise ValueError("Target variable for classification algorithms must be integer encoded.")
        x = self.data.iloc[:, np.isin(self.data.columns, [self.info["TARGET_COLUMN"]] +
                                      self.info["STRATIFICATION_COLUMN"], invert=True)]
        self.problem_specs = self.infer_problem_specs(x, y)

        # Generalization Assessment
        if self.info["STRATIFICATION_COLUMN"][0] == '':  # TODO: fix this check with a more robust one
            strat_df = None
        else:
            strat_df = self.data[self.info["STRATIFICATION_COLUMN"]]
        cv = self.generate_folds(x, y, stratification=strat_df)
        self.validation, self.predictions = self.generalization_assessment(x, y, cv=cv)

        # Fit model on entire train data
        if self.problem_specs["type"] == "classification" and not self.problem_specs["balanced"]:
            rus = RandomUnderSampler(sampling_strategy='not minority', replacement=False)
            resampled_index, _ = rus.fit_resample(x.index.values.reshape(-1, 1), y)
            resampled_index = resampled_index.flatten()
            self.trained_model = self.fit(x.loc[resampled_index, :], y.loc[resampled_index])
        else:
            self.trained_model = self.fit(x, y)

        # Export
        unique_id = self.generate_unique_id()
        self.export_trained_model(unique_id)
        self.export_predictions(unique_id)
        self.export_validation(unique_id)

        # Clean-up
        rmtree(self.local_path)

        # Report Loss to Hypertune
        if self.hypertune_loss is None or self.validation is None:
            return
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=self.hypertune_loss,
            metric_value=self.validation['test_' + HYPERTUNE_LOSSES[self.hypertune_loss]].mean())
