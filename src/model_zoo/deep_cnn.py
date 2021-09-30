import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl

def get_model(
        input_shape=(200,4),
        num_outputs=1,
        bn=False,
        dropout1=None,
        dropout2=None,
        kernel_regularizer=None,
        activation='relu',
        name="model",
        factor=1,
        logits_only=False,
            ):
    assert isinstance(factor, int) and factor > 0
    x = tfkl.Input(input_shape, name='input')

    # 1st conv block
    y = tfkl.Conv1D(
            filters=24*factor, kernel_size=19, padding='same',
            kernel_regularizer=kernel_regularizer)(x)
    if bn:
        y = tfkl.BatchNormalization()(y)
    y = tfkl.Activation(activation)(y)
    if dropout1:
        y = tfkl.Dropout(rate=dropout1)(y)

    # 2nd conv block
    y = tfkl.Conv1D(
            filters=32*factor, kernel_size=7, padding='same',
            kernel_regularizer=kernel_regularizer)(y)
    if bn:
        y = tfkl.BatchNormalization()(y)
    y = tfkl.Activation('relu')(y)
    if dropout1:
        y = tfkl.Dropout(rate=dropout1)(y)
    y = tfkl.MaxPool1D(pool_size=4)(y)

    # 3rd conv block
    y = tfkl.Conv1D(
            filters=48*factor, kernel_size=5, padding='same',
            kernel_regularizer=kernel_regularizer)(y)
    if bn:
        y = tfkl.BatchNormalization()(y)
    y = tfkl.Activation('relu')(y)
    if dropout1:
        y = tfkl.Dropout(rate=dropout1)(y)
    y = tfkl.MaxPool1D(pool_size=4)(y)

    # 4th conv block
    y = tfkl.Conv1D(
            filters=64*factor, kernel_size=5, padding='same',
            kernel_regularizer=kernel_regularizer)(y)
    if bn:
        y = tfkl.BatchNormalization()(y)
    y = tfkl.Activation('relu')(y)
    if dropout1:
        y = tfkl.Dropout(rate=dropout1)(y)
    y = tfkl.MaxPool1D(pool_size=3)(y)

    # FC block
    y = tfkl.Flatten()(y)
    y = tfkl.Dense(96*factor, kernel_regularizer=kernel_regularizer)(y)
    if bn:
        y = tfkl.BatchNormalization()(y)
    y = tfkl.Activation('relu')(y)
    if dropout2:
        y = tfkl.Dropout(rate=dropout2)(y)

    # final FC block
    y = tfkl.Dense(1, kernel_regularizer=kernel_regularizer, name='logits')(y)
    if logits_only:
        return tfk.Model(x,y,name=name)
    else:
        y = tfkl.Activation('sigmoid', name='probabilities')(y)
        return tfk.Model(x,y,name=name)
