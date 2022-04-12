import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow_probability import distributions as tfd
from .augmentations import Augmentation

class _Model(tfk.Model):
    """
    Useful base class when extending the keras.Model class.
    """
    def __init__(self, model, training=True, name='custom_model'):
        super().__init__(name=name)
        self._model = model
        self._training = training
    
    def summary(self):
        return self.model.summary()
    
    @property
    def training(self):
        return self._training
    
    @training.setter
    def training(self, t):
        assert t in [True, False]
        self._training = t
        
    @property
    def layers(self):
        return self.model.layers

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        self._model = m
    
    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)
        super().compile(*args, **kwargs)

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
    
    def train_step(self, data):
        x, y = data

        # record differentiable ops
        with tf.GradientTape() as tape:
            # make predictions
            # the training flag is to enable turning bn/dropout on or off
            y_pred = self.model(x, training=self.training)
            
            # compute the total loss 
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients and take optimization step
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # finish up
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x,y = data
        ypred = self(x, training=False)
        loss = self.compiled_loss(y, ypred, regularization_losses=None)
        self.compiled_metrics.update_state(y, ypred)
        res = {m.name:m.result() for m in self.metrics}
        return res

class ModelWrapper(_Model):
    """
    Wraps a standard keras model to add custom functionality 
    implemented in the _Model class.
    """
    def __init__(self, model, name='model'):
        super().__init__(model=model, name=name)
    
class AdversariallyTrainedModel(_Model):
    @property 
    def clean(self):
        return self._clean
    
    @clean.setter
    def clean(self, c):
        self._clean = c
    
    @property
    def mix(self):
        return self._mix 
    
    @mix.setter 
    def mix(self, m):
        self._mix = m
    
    def __init__(self, model, training=True, attacker='pgd', attacker_kwargs=None, name="adversarial_model"):
        super().__init__(model, training, name)
        self.attacker_name = attacker
        if attacker_kwargs is None:
            if attacker == 'pgd':
                attacker_kwargs = {
                                'learning_rate':1e-2, 
                                'epsilon':0.05, 
                                'num_steps':10, 
                                'grad_sign':True, 
                                'decay':False
                                    }
            else:
                raise ValueError("Only PGD attacker defaults available currently.")
        self.attacker_kwargs = attacker_kwargs
        self._clean = False 
        self._mix = True
    
    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.attacker = ATTACKERS[self.attacker_name](
                                            model=self.model, 
                                            loss=self.model.compiled_loss._losses,
                                            **self.attacker_kwargs
                                                    )
    
    def train_step(self, data):
        x, y = data
        if not self.clean:
            x_attack = self.attacker.generate(data)
            if self.mix:
                x = tf.concat([x_attack, x], axis=0)
                y = tf.concat([y,y], axis=0)
            else:
                x = x_attack
        return super().train_step((x, y))

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
            ypredlogits = self(x, training=self.training)
            ypred = self.last_layer_activation(ypredlogits)

            delta = 0.1*tf.random.normal(shape=tf.shape(x))
            x_noise = x + delta
            y_noise = self.model(x_noise, training=True)
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
    def _check_augmentation_validity(self, augmentations):
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
    """
    Manifold mixup is a generalization of the input mixup regularization scheme where 
    intermediate layer representations are mixed up rather than just the input layer
    """
    def __init__(self, model, ks, alpha=1., name="mixup_model"):
        """
        ks -> List of layer activation indices to use for manifold mixup.
        """
        super().__init__(model=model, name=name)
        self.lam_dist = tfd.Beta(alpha, alpha)
        self.ks = ks
    
    def fit(self, *args, **kwargs):
        if not self.run_eagerly:
            print("WARNING: This model can only be run in eager mode.Switching run_eagerly property...")
            self.run_eagerly = False
        return super().fit(*args, **kwargs)

    def train_step(self, data):
        x, y = data
        k = np.random.randint(0, len(self.ks)) ## pick a random layer index
        lam = self.lam_dist.sample()  # sample a mixing coefficient 
        idxs = tf.random.shuffle(tf.range(tf.shape(x)[0])) # shuffled indices 
        xp, yp = tf.gather(x, idxs, axis=0), tf.gather(y, idxs, axis=0) ## shuffled batch
        ym = lam*y + (1-lam)*yp ## mixup the labels

        # record differentiable ops
        with tf.GradientTape() as tape:
            y_pred, ym_pred = x, xp
            for i, layer in enumerate(self.model.layers):
                y_pred = layer(y_pred, training=self.training)
                ym_pred = layer(ym_pred, training=self.training)
                if i == k:
                    ym_pred = lam*y_pred + (1-lam)*ym_pred # mixup the representation at the kth layer
            
            # stack the original minibatch to create augmented minibatch and calculate loss
            y_aug = tf.concat([y, ym], axis=0)
            y_pred_aug = tf.concat([y_pred, ym_pred], axis=0)
            loss = self.compiled_loss(y_aug, y_pred_aug, regularization_losses=self.losses)

        # Compute gradients and take optimization step
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # finish up
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

class ManifoldGaussianNoiseModel(_Model):
    """
    Manifold Gaussian noise is a generalization of the input gaussian noise regularization scheme where 
    intermediate layer representations are mixed up rather than just the input layer
    """
    def __init__(self, model, stddev=0.1, name="mixup_model",):
        """
        ks -> List of layer activation indices to use for manifold mixup.
        """
        super().__init__(model=model, name=name)
        self.noise_dist = tfd.Normal(loc=0., scale=stddev)
        ks = [0]
        for i, layer in enumerate(model.layers):
            if 'activation' in layer.name:
                if layer.activation.__name__ != 'linear' and layer.activation.__name__ != 'sigmoid':
                    ks.append(i)
        self.ks = ks
    
    def fit(self, *args, **kwargs):
        if not self.run_eagerly:
            print("WARNING: This model can only be run in eager mode. Switching run_eagerly property...")
            self.run_eagerly = True
        return super().fit(*args, **kwargs)

    def train_step(self, data):
        x, y = data
        k = np.random.randint(0, len(self.ks)) ## pick a random layer index
        ym = y ## augmented data labels 

        # record differentiable ops
        with tf.GradientTape() as tape:
            y_pred, ym_pred = x, x
            for i, layer in enumerate(self.model.layers):
                y_pred = layer(y_pred, training=self.training)
                ym_pred = layer(ym_pred, training=self.training)
                if i == k:
                    ym_pred = ym_pred + self.noise_dist.sample(tf.shape(ym_pred)) # add noise to the representation at the kth layer
            
            # stack the original minibatch to create augmented minibatch and calculate loss
            y_aug = tf.concat([y, ym], axis=0)
            y_pred_aug = tf.concat([y_pred, ym_pred], axis=0)
            loss = self.compiled_loss(y_aug, y_pred_aug, regularization_losses=self.losses)

        # Compute gradients and take optimization step
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # finish up
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}