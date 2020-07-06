from sklearn.model_selection import StratifiedKFold, KFold
from imblearn.under_sampling import RandomUnderSampler
from abc import ABC, abstractmethod
from pandas import concat, DataFrame
import numpy as np


class BaseValidationSchema(ABC):

    def __init__(self, schema_generator_class=None, params=None):
        self.schema_generator = schema_generator_class(**params)

    def split(self, x, y, **kwargs):
        self.customize_schema_generator(x, y, **kwargs)
        for train_index, validation_index in self._split(x, y, **kwargs):
            if kwargs['balanced'] is True or kwargs['balanced'] is None:
                yield train_index, validation_index
            else:
                # undersample train data
                rus = RandomUnderSampler(sampling_strategy='not minority', replacement=False)
                resampled_train_index, _ = rus.fit_resample(x.index[train_index].values.reshape(-1, 1),
                                                            y.iloc[train_index])
                resampled_mask = [item in resampled_train_index for item in x.index]
                resampled_train_rows = [i for i, item in enumerate(resampled_mask) if item is True]
                yield np.asarray(resampled_train_rows), validation_index

    def customize_schema_generator(self, x, y, **kwargs):
        pass

    @abstractmethod
    def _split(self, x, y, **kwargs):
        pass


class KFoldValidationSchema(BaseValidationSchema):

    def __init__(self, params=None):
        super().__init__(schema_generator_class=KFold, params=params)

    def _split(self, x, y, **kwargs):
        yield self.schema_generator.split(x, y)


class StratifiedKFoldValidationSchema(BaseValidationSchema):

    def __init__(self, params=None):
        super().__init__(schema_generator_class=StratifiedKFold, params=params)

    def _split(self, x, y, **kwargs):
        for train, validation in self.schema_generator.split(x, kwargs['stratification_variable']):
            yield train, validation


class GroupStrataKFoldValidationSchema(BaseValidationSchema):
    """
        Use stratification variable unique values to form groups on which you perform KFold. This ensures that each
        group will appear entirely either in train or validation set.
    """

    def __init__(self, params=None):
        super().__init__(schema_generator_class=KFold, params=params)

    def _split(self, x, y, **kwargs):

        groups = np.unique(kwargs['stratification_variable'])
        # sample_list = []
        for train_groups, validation_groups in self.schema_generator.split(groups):
            df_list = []
            # print("")
            # print("Training on groups:")
            # for item in train_groups:
            #     print(groups[item])
            # print("Validation on groups:")
            # for item in validation_groups:
            #     print(groups[item])
            for value in validation_groups:
                    df_list.append(kwargs['stratification_variable'] == groups[value])
            validation_indexes = DataFrame(concat(df_list, axis=1).sum(axis=1), columns=['flag'])
            train_indexes = DataFrame(1-validation_indexes)
            validation_indexes['integer_based_index'] = range(x.shape[0])
            train_indexes['integer_based_index'] = range(x.shape[0])

            # if not sample_list:
            #     sample_list = list(x.sample(5).index)
            # for item in sample_list:
            #     print("Sample {}, belonging to {} is in validation group: {}".format(item,
            #                                                                          kwargs['stratification_variable'].loc[item].values[0],
            #                                                                          validation_indexes.loc[item]))

            yield train_indexes['integer_based_index'].loc[train_indexes['flag'] == 1].values, validation_indexes['integer_based_index'].loc[validation_indexes['flag'] == 1].values