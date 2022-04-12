import numpy as np
import tensorflow as tf

@tf.function
def input_grad_batch(X, y, model, loss):
  """fast function to generate saliency maps"""
  if not tf.is_tensor(X):
    X = tf.Variable(X)

  with tf.GradientTape() as tape:
    tape.watch(X)
    predictions = model(X)
    loss_value = loss(y, predictions)
    return tape.gradient(loss_value, X) 

class AdversarialAttack():
    def __init__(self, model, loss, epsilon=0.1):
        self.model = model
        self.loss = loss
        self.epsilon = epsilon
    
    def generate(self, data):
        raise NotImplementedError("Implement this method in the subclass.")

class PGDAttack(AdversarialAttack):
    """
    Projected gradient descent attack.
    """
    def __init__(self, model, loss, learning_rate=0.1, epsilon=0.1, num_steps=10, grad_sign=True, decay=False):
        super().__init__(model, loss, epsilon)
        self.learning_rate = tf.Variable(learning_rate, trainable=False)
        self.grad_sign = grad_sign
        self.num_steps = num_steps
        self.decay = decay

    def generate(self, data):
        x, y = data
        x_pgd = tf.identity(x)
        for i in range(self.num_steps):
            delta = input_grad_batch(x_pgd, y, self.model, self.loss)

            # convert gradient to a sign (works better than pure gradients)
            if self.grad_sign:
                delta = tf.math.sign(delta)  

            # decay learning rate
            if self.decay:
                learning_rate = self.learning_rate/(i+10)
            else:
                learning_rate = self.learning_rate

            # update inputs   
            x_pgd += learning_rate*delta     

            # clip so as to project onto max perturbation of epsilon
            x_pgd = tf.clip_by_value(x_pgd, x-self.epsilon, x+self.epsilon)

        return x_pgd

class FGSMAttack(AdversarialAttack):
    """
    fast Gradient sign attack. 
    """
    def __init__(self, model, loss, epsilon=0.1):
        super().__init__(model, loss, epsilon)

    def generate(self, x, y):
        delta = input_grad_batch(x, y, self.model, self.loss)
        return x + self.epsilon*tf.math.sign(delta)
