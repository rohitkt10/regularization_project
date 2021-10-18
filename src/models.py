import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from .augmentations import Augmentation

class _Model(tfk.Model):
    """
    Useful base class when extending the keras.Model class.
    """
    def __init__(self, model, name='custom_model'):
        super().__init__(name=name)
        self._model = model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        self._model = m

    def save(self, *args, **kwargs):
        return self.model.save(*args, **kwargs)

    def save_weights(self, *args, **kwargs):
        return self.model.save_weights(*args, **kwargs)

    def get_weights(self,):
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)

    def load_weights(self, *args, **kwargs):
        return self.model.load_weights(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def call(self, *args, **kwargs):
        return self.model.call(*args, **kwargs)

    def test_step(self, data):
        x,y = data
        ypred = self(x, training=False)
        loss = self.compiled_loss(y, ypred, regularization_losses=None)
        self.compiled_metrics.update_state(y, ypred)
        res = {m.name:m.result() for m in self.metrics}
        return res


class JacobianRegularizedModel(_Model):
    """
    Regularize the model by adding the squared Frobenius norm
    of the output logits wrt to the inputs.
    """

    def __init__(self, model, name='jacobian_regularized_model'):
        super().__init__(model=model, name=name)

    def predict(self, *args, **kwargs):
        return self.last_layer_activation(super().predict(*args, **kwargs)).numpy()

    def compile(self, beta=1., last_layer_activation='sigmoid', **kwargs):
        """
        PARAMETERS:

        1. beta <float> - The Jacobian regularization coefficient.
        2. last_layer_activation <callable> - The activation function to be applied to the model logits.
        3. kwargs - Additional parameters to be sent to keras.Model.compile.
        """
        assert beta > 0., "Jacobian regularization coefficient must be positive."
        self.beta = tf.Variable(beta, dtype=tf.float32, trainable=False)
        self.last_layer_activation = tfk.layers.Activation(last_layer_activation)
        self.model.compile(**kwargs)
        super().compile(**kwargs)

    def train_step(self, data):
        x, y = data
        batchsize = tf.shape(x)[0]

        # differentiable operations
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            ypredlogits = self(x, training=True)
            ypred = self.last_layer_activation(ypredlogits)

            delta = 0.1*tf.random.normal(shape=tf.shape(x))
            x_noise = x + delta
            y_noise = self(x_noise, training=True)
            J = tape.batch_jacobian(y_noise, x_noise)  ## (batch, numoutputs, L, A)
            J = tf.reshape(J, (tf.shape(J)[0], -1)) ## (batch, numoutputs*L*A)
            Jfro2 = tf.linalg.norm(J, axis=1)**2  ## (batch,)
            Jfro2 = tf.reduce_mean(Jfro2, axis=0)  ## scalar
            loss = self.compiled_loss(y, ypred, regularization_losses=self.losses+[self.beta*Jfro2])

        # gradient step
        trainable_variables = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        # update metrics
        self.compiled_metrics.update_state(y, ypred)

        res = {m.name:m.result() for m in self.metrics}
        res['sq_jacobian_norm'] = Jfro2
        return res

    def test_step(self, data):
        x,y = data
        ypred = self.last_layer_activation(self(x, training=False))
        loss = self.compiled_loss(y, ypred, regularization_losses=None)
        self.compiled_metrics.update_state(y, ypred)
        res = {m.name:m.result() for m in self.metrics}
        return res

class AugmentedModel(_Model):
    def _check_augmentation_validity(augmentations):
        #res = [isinstance(augmentation, Augmentation) for augmentation in augmentations]
        #assert np.all(res)
        assert all(isinstance(aug, Augmentation) for aug in augmentations)

    def __init__(self, model, augmentations, subsample=None, name="augmented_model"):
        """
        PARAMETERS:
        1. model <tfk.Model> - Input model.
        2. augmentations <list of Augmentations> - A list of objects of type Augmentations.
        3. subsample <int> - Number of samples to retain in the augmented data. By default,
                             we subsample to the same number as the batch size.
        """
        self._check_augmentation_validity(augmentations)
        super().__init__(model=model, name=name)
        self.augmentations = augmentations
        self.subsample = subsample

    def _get_augmented_dataset(self, data):
        batchsize = tf.shape(data[0])[0]

        augmented_x = []
        augmented_y = []
        for augmentation in self.augmentations:
            (_x, _y) = augmentation(data)
            augmented_x.append(_x)
            augmented_y.append(_y)
        _x = tf.concat(augmented_x, axis=0)
        _y = tf.concat(augmented_y, axis=0)
        augmented_batchsize = tf.cast(tf.shape(_x)[0], tf.float32)

        # subsample the augmented dataset
        if not self.subsample:
            idx = tf.random.uniform((batchsize,), 0, augmented_batchsize,)
        else:
            idx = tf.random.uniform((self.subsample,), 0, augmented_batchsize)
        idx = tf.cast(tf.math.floor(idx), tf.int32)
        _x, _y = tf.gather(_x, idx, axis=0), tf.gather(_y, idx, axis=0)

        data = (_x, _y)
        return data

    def train_step(self, data):
        data = self._get_augmented_dataset(data)
        return super().train_step(data)

class ManifoldMixupModel(_Model):
    def __init__(self, model, ks, alpha=1.,name="mixup_model"):
        """
        ks -> List of layer activation indices to use for manifold mixup.
        """
        super().__init__(name=name)
        self.model = model
        self.lam_dist = tfd.Beta(alpha, alpha)
        self.ks = ks
    
    def call(self, *args, **kwargs):
        return self.model.call(*args, **kwargs)

    def train_step(self, data):
        x, y = data
        k = np.random.randint(0, len(self.ks)) ## pick a random layer index
        lam = self.lam_dist.sample()
        idxs = tf.random.shuffle(tf.range(tf.shape(x)[0])) # shuffled indices 
        xp, yp = tf.gather(x, idxs, axis=0), tf.gather(y, idxs, axis=0) ## shuffled batch
        ym = lam*y + (1-lam)*yp ## mixup the labels

        # record differentiable ops
        with tf.GradientTape() as tape:
            y_pred, ym_pred = x, xp
            for i, layer in enumerate(self.model.layers):
                y_pred = layer(y_pred, training=True)
                ym_pred = layer(ym_pred, training=True)
                if i == k:
                    ym_pred = lam*y_pred + (1-lam)*ym_pred # mixup the representation at the kth layer
            
            # stack the original minibatch with the augmented data 
            y = tf.concat([y, ym], axis=0)
            y_pred = tf.concat([y_pred, ym_pred], axis=0)
            
            # calcalate total loss 
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # take optimization step 
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)

        # return dictionary of metrics 
        return {m.name: m.result() for m in self.metrics}