import lightgbm
import numpy as np
from jax import grad, vmap
import jax.numpy as jnp
from jax.scipy import special
from sklearn.preprocessing import LabelBinarizer


class CustomLoss:
    """
    train_data must be an object of type lgbm.Dataset
    """

    def __init__(self, name=None, higher_is_better=False):
        self.name = name
        self.higher_is_better = higher_is_better
        self._grad_fun = grad(self.loss, argnums=0)
        self._hess_fun = vmap(grad(grad(self.loss, argnums=0), argnums=0))

    def loss(self, preds, train_data):
        # Should implement all transformations and numerical preprocessing
        # to ensure correct derivatives
        raise NotImplementedError

    @staticmethod
    def _preprocess_preds(preds):
        return preds

    @staticmethod
    def _preprocess_train_data(train_data):
        return train_data

    def preprocess(self, preds, train_data):
        return self._preprocess_preds(preds), self._preprocess_train_data(train_data)

    def _grad(self, preds, train_data, normalize=False):
        if normalize:
            return self._grad_fun(preds, train_data)._value / jnp.linalg.norm(self._grad_fun(preds, train_data)._value)
        return self._grad_fun(preds, train_data)._value

    def _hess(self, preds, train_data, normalize=False):
        # Works iff Jacobian in diagonal and returns the diagonal of the hessian matrix.
        # If the Jacobian is non-diagonal, this returns the sum of the rows of the hessian matrix.
        # https://github.com/google/jax/issues/564
        if normalize:
            return self._hess_fun(preds, train_data)._value / jnp.linalg.norm(self._hess_fun(preds, train_data)._value)
        return self._hess_fun(preds, train_data)._value

    def objective(self, preds, train_data, normalize=False):
        preprocessed_preds, preprocessed_train_data = self.preprocess(preds, train_data)
        return self._grad(preprocessed_preds, preprocessed_train_data, normalize), \
               self._hess(preprocessed_preds, preprocessed_train_data, normalize)

    def eval(self, preds, train_data):
        preprocessed_preds, preprocessed_train_data = self.preprocess(preds, train_data)
        return self.name, self.loss(preprocessed_preds, preprocessed_train_data), \
               self.higher_is_better

class MyCustomLoss(CustomLoss):
    # sklearn API uses (y_true, y_pred) signature while PythonAPI is more flexible
    # and allows to pass the entire train data to compute loss function.
    # TODO: generalize from sklearnAPI to PythonAPI
    def __init__(self):
        super().__init__(name='MyCustomLoss', higher_is_better=False)

    def __call__(self, preds, train_data):
        return sum((preds - train_data.get_label()) ** 2)

    def grad(self, preds, train_data, normalize=False):
        # return -1 + 2*(y_pred - y_true > 0)
        if normalize:
            return 2*(preds - train_data.get_label()) / np.linalg.norm(2*(preds - train_data.get_label()))
        return 2*(preds - train_data.get_label())

    def hess(self, preds, train_data, normalize=False):
        if normalize:
            return 2*np.ones(len(preds)) / np.linalg.norm(2*np.ones(len(preds)))
        return 2*np.ones(len(preds))

class CustomMSE(CustomLoss):

    def __init__(self):
        super().__init__(name='CustomMSE', higher_is_better=False)

    @staticmethod
    def _preprocess_train_data(train_data):
        if isinstance(train_data, lightgbm.Dataset):
            return train_data.get_label()
        else:
            return train_data

    def loss(self, preds, train_data):
        return jnp.sum((preds - train_data) ** 2)

class CustomMSEVarianceDecay(CustomLoss):

    def __init__(self):
        super().__init__(name='CustomMSEVarianceDecay', higher_is_better=False)

    @staticmethod
    def _preprocess_train_data(train_data):
        if isinstance(train_data, lightgbm.Dataset):
            return train_data.get_label()
        else:
            return train_data

    def loss(self, preds, train_data):
        # MSE with an additional penalty on how disperse errors are
        return jnp.sum((preds - train_data) ** 2) * (1+jnp.var((preds - train_data) ** 2))

class MyCustomLogLoss(CustomLoss):
    def __init__(self):
        super().__init__(name='MyCustomLogLoss', higher_is_better=False)

    def __call__(self, preds, train_data):
        train_data = train_data.get_label()
        preds = special.expit(preds)

        #ll = np.empty_like(p)
        #pos = train_data == 1
        #ll[pos] = np.log(p[pos])
        #ll[~pos] = np.log(1 - p[~pos])

        ll = jnp.log(preds) * train_data + jnp.log(1 - preds) * (1-train_data)
        return -ll.mean()

    def grad(self, preds, train_data):
        # return -1 + 2*(y_pred - y_true > 0)
        p = special.expit(preds)
        label = train_data.get_label()
        return (p - label)/len(preds) # account for scaling 1/N

    def hess(self, preds, train_data):
        p = special.expit(preds)
        return p * (1-p)

class LogLossVarianceDecay(CustomLoss):

    def __init__(self):
        super().__init__(name='LogLoss_VarianceDecay', higher_is_better=False)

    @staticmethod
    def _preprocess_train_data(train_data):
        if isinstance(train_data, lightgbm.Dataset):
            return train_data.get_label()
        else:
            return train_data

    def loss(self, preds, y_true):
        # train_data here are class labels
        return -jnp.mean(jnp.log(special.expit(preds)) * y_true +
                         jnp.log(1 - special.expit(preds)) * (1-y_true))
        #return -jnp.mean(jnp.log(jnp.sum(preds * train_data, axis=1)))

class VectorLogLossVarianceDecay(CustomLoss):
    """
    FROM LGBM DOCS
    For multi-class task, the preds is group by class_id first, then group by row_id.
    If you want to get i-th row preds in j-th class, the access way is score[j * num_data + i] and you should group grad and hess in this way as well.

    In the multi-class case the expit fuction is replaced with the softmax function. This is a different functional form that yields different gradients.
    In particular softmax ties all probabilities from different classes together so that moving a single one generates a non-zero gradient in all others.
    """

    def __init__(self):
        super().__init__(name='VectorLogLoss_VarianceDecay', higher_is_better=False)

    @staticmethod
    def _preprocess_train_data(train_data):
        if isinstance(train_data, lightgbm.Dataset):
            label = train_data.get_label()
        else:
            label = train_data
        if len(np.unique(label)) == 2:
            tmp = np.zeros((len(label), 2))
            tmp[:, 0] = label
            tmp[:, 1] = 1 - label
            return tmp
        else:
            lb = LabelBinarizer()
            return lb.fit_transform(label)

    @staticmethod
    def _preprocess_preds(preds):
        if len(preds.shape) == 1:
            tmp = np.zeros((len(preds), 2))
            tmp[:, 0] = preds
            tmp[:, 1] = 1 - preds
            return tmp
        return preds

    def loss(self, preds, y_true):
        # train_data here are class labels
        #tmp_loss = softmax(preds) * y_true --> using softmax here produces opposite
        # gradients in different dimensions. However not using it here takes the wrong derivative w.r.t. model scores.
        tmp_loss = preds * y_true
        try:
            return -jnp.mean(jnp.log(jnp.sum(tmp_loss, axis=1)))
        except:
            pass
            # return -jnp.mean(jnp.log(jnp.sum(tmp_loss.primal.primal.val, axis=1)))
