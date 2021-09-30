import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow import keras as tfk

class Augmentation():
    def __call__(self, data):
        raise NotImplementedError()

class MixupAugmentation(Augmentation):
    def __init__(self, alpha=0.2):
        self.lam_dist = tfd.Beta(alpha, alpha)

    def _einsum_exp(self, x):
        n = len(x.shape)
        if n == 1:
            return "i,i -> i"
        elif n == 2:
            return "ij, i -> ij"
        elif n == 3:
            return "ijk, i -> ijk"
        elif n == 4:
            return "ijkl, i -> ijkl"
        else:
            raise NotImplementedError()

    def __call__(self, data):
        x, y = data
        idxs = tf.random.shuffle(tf.range(tf.shape(x)[0])) ## (batch,)
        xp, yp = tf.gather(x, idxs, axis=0), tf.gather(y, idxs, axis=0)  ## (permutation of data batch)
        lam = self.lam_dist.sample(tf.shape(x)[0])  ## (batch,)
        expx = self._einsum_exp(x)  ## string
        expy = self._einsum_exp(y)  ## string
        xm = tf.einsum(expx, x, lam) + tf.einsum(expx, x, 1.-lam) ## (same shape as x)
        ym = tf.einsum(expy, y, lam) + tf.einsum(expy, y, 1.-lam) ## (same shape as y)
        return (xm, ym)

class GaussianNoiseAugmentation(Augmentation):
    def __init__(self, mean=0., stddev=1e-2,):
        self.gdist = tfd.Normal(loc=mean, scale=stddev)

    def __call__(self, data):
        x, y = data
        delta = self.gdist.sample(tf.shape(x))
        xd = x + delta
        return (xd, y)

class RCAugmentation(Augmentation):
    def __call__(self, data):
        x, y = data
        x_rc = x[:, ::-1, ::-1]
        return (x_rc, y)
