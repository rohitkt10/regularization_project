import numpy as np, os, h5py, sys, pandas as pd, argparse

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
from tensorflow.keras import callbacks as tfkc
from tensorflow_probability import distributions as tfd

import tfomics
from tfomics import impress, explain, evaluate

from train_model_utils import (models, BASERESULTSDIR, SYNTHETIC_DATADIR)
from src.utils import _get_synthetic_data
from src.callbacks import ExtendedEarlyStopping
from src.models import ManifoldGaussianNoiseModel

# parameters 
BATCHSIZE = 128
EPOCHS = 100

# define the model
def deep(
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
get_model = {'deep':deep}

## Training
def train(model, ckptdir, traindata, validdata, testdata, model_test, fit_verbose='auto'):
    # create ckpt directory if it doesnt exist
    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)

    # set up compile options and compile the model
    acc = tfk.metrics.BinaryAccuracy(name='acc')
    auroc = tfk.metrics.AUC(curve='ROC', name='auroc')
    aupr = tfk.metrics.AUC(curve='PR', name='aupr')
    optimizer = tfk.optimizers.Adam(learning_rate=1e-3)
    loss = tfk.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
    model.compile(optimizer=optimizer, loss=loss, metrics=[acc, auroc, aupr], )

    ## set up callbacks
    callbacks = []
    csvlogger = tfk.callbacks.CSVLogger(os.path.join(ckptdir, "log.csv"))
    callbacks.append(csvlogger) 
    es_callback = ExtendedEarlyStopping(
                                    EPOCHS,
                                    testdata, 
                                    model_test,
                                    ckptdir,
                                    threshold=0.1,
                                    top_k=10,
                                    monitor='val_auroc',
                                    patience=10,
                                    verbose=1,
                                    mode='max',
                                    restore_best_weights=True
                                        )
    callbacks.append(es_callback)

    # fit the model
    model.fit(
        traindata.shuffle(10000).batch(BATCHSIZE),
        validation_data=validdata.batch(64),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=fit_verbose,
            )
    
    print("Calculating test classification metrics ...")
    testres = model.evaluate(testdata[0], testdata[1], return_dict=True)
    columns = list(testres.keys())
    data = [list(testres.values())]
    df = pd.DataFrame(columns=columns, data=data)
    df.to_csv(os.path.join(ckptdir, 'testmetrics.csv'))

def main():
    # get keyboard arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", default='relu', help='First layer activation.', type=str)
    parser.add_argument("--bn", action="store_true", help="Whether to use batch norm",)
    parser.add_argument("--nobn", action="store_true", help="Whether NOT to use batch norm",)
    args = parser.parse_args()
    
    # load data 
    traindata, validdata, testdata, model_test = _get_synthetic_data(SYNTHETIC_DATADIR)
    traindata = tf.data.Dataset.from_tensor_slices(traindata)
    validdata = tf.data.Dataset.from_tensor_slices(validdata)

    # training settings 
    bn = args.bn
    if bn:
        BN = 'bn'
    else:
        BN = 'nobn'
    activation = args.activation
    kernel_regularizer=None
    CKPTDIR = os.path.join(BASERESULTSDIR, 'table_1', BN, activation, 'manifold_gn')
    if not os.path.exists(CKPTDIR):
        os.makedirs(CKPTDIR)
    
    # do several trials
    numtrials = 10
    for trial in range(numtrials):
        print("*************")
        print(f"Trial : {trial+1}")
        print("*************")
        for factor in [1, 4, 8]:
            print(f"Model : deep - {factor}")
            print('----------------------')
            model = get_model['deep'](
                                    bn=bn,
                                    dropout1=0.1,
                                    dropout2=0.5,
                                    activation=activation, 
                                    factor=factor,
                                    kernel_regularizer=kernel_regularizer,
                                    )
            ks = [0]
            for i, layer in enumerate(model.layers):
                if 'activation' in layer.name:
                    if layer.activation.__name__ != 'linear' and layer.activation.__name__ != 'sigmoid':
                        ks.append(i)
            model = ManifoldGaussianNoiseModel(model=model, ks=ks, stddev=0.15)
            ckptdir = os.path.join(CKPTDIR, f"deep_{factor}", f"trial_{trial+1:02d}")
            if os.path.exists(ckptdir):
                # dont retrain if results already exist
                continue
            train(model, ckptdir, traindata, validdata, testdata, model_test, fit_verbose=0)
            print('----------------------')
    

if __name__=='__main__':
    main()