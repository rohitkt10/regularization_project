import tensorflow as tf
from tensorflow import keras as tfk


class JacobianRegularizedModel(tfk.Model):
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
        return self.last_layer_activation(self.model.predict(*args, **kwargs)).numpy()

    def call(self, *args, **kwargs):
        return self.model.call(*args, **kwargs)

    def __init__(self, model, name='jacobian_regularized_model'):
        super().__init__(name=name)
        self._model = model

    def compile(self, optimizer, loss=None, metrics=None, beta=1., last_layer_activation='sigmoid', *args, **kwargs):
        assert beta > 0., "Jacobian regularization coefficient must be positive."
        self.model.compile(metrics=metrics, loss=loss)
        super().compile(loss=loss, metrics=metrics, optimizer=optimizer, *args, **kwargs)
        self.beta = tf.Variable(beta, dtype=tf.float32, trainable=False)
        self.last_layer_activation = tfk.layers.Activation(last_layer_activation)

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
        res['jacobian_norm'] = Jfro2
        return res

    def test_step(self, data):
        x,y = data
        ypred = self.last_layer_activation(self(x, training=False))
        loss = self.compiled_loss(y, ypred, regularization_losses=None)
        self.compiled_metrics.update_state(y, ypred)
        res = {m.name:m.result() for m in self.metrics}
        return res
