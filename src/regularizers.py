import tensorflow as tf
from tensorflow import keras as tfk

class SpectralNormRegularizer(tfk.regularizers.Regularizer):
    """
    Regularize the spectral norm (i.e. largest singular value).
    """
    def __init__(self, lmbda=1., n_iter=10, *args, **kwargs):
        """
        lmbda -> Regularization constant.
        n_iter -> Number of iterations of the power method.
        """
        assert lmbda > 0.
        super().__init__(*args, **kwargs)
        self.lmbda = lmbda
        self.n_iter = 10

    def _power_iteration(self, W):
        """
        Power method for computing largest singular value.
        """
        _W = tf.einsum("ij, ik ->jk", W, W)
        x = tf.random.normal((tf.shape(W)[1], 1))
        for _ in range(self.n_iter):
            x = tf.matmul(_W, x)
        v = x / tf.linalg.norm(x)
        prod = tf.matmul(W, v)
        sigma = tf.linalg.norm(prod)
        u = prod / sigma
        return u,v,sigma

    def __call__(self, x):
        x = tf.reshape(x, (-1, tf.shape(x)[-1]))
        u,v,sigma = self._power_iteration(x)
        return self.lmbda * sigma**2

    def get_config(self):
        return {'lmbda':self.lmbda, 'n_iter':self.n_iter}

class RegularizersContainer:
    def __init__(self, *regularizers):
        """
        regularizers -> A list of callables to be applied during regularization.
        """
        self._regularizers = list(regularizers)

    @property
    def regularizers(self):
        return self._regularizers

    def append(self, regularizer):
        self._regularizers.append(regularizer)

    def __call__(self, x):
        reg = 0.0
        for regularizer in self.regularizers:
            reg = reg + regularizer(x)
        return reg
