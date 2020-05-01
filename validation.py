from sklearn.model_selection import StratifiedKFold, KFold
from imblearn.under_sampling import RandomUnderSampler
from abc import ABC, abstractmethod
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