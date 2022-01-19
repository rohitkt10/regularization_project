"""
Task 5 - Comparing the classification performance and interpretability of
shallow and deep CNN models under various regularization settings.
-------------------------------------------------------------------

Subtask 5.6 - Table 2 spectral norm regularization training results

In this subtask we train the shallow and deep CNN with batch normalization
dropout. The test is conducted for 3 versions of the shallow and deep
CNNs each - scaling each of these types of models by a factor of 1, 4 and 8.
"""

import numpy as np, os, sys, h5py, pandas as pd, argparse
from pdb import set_trace as keyboard

sys.path.append("..")
from src.callbacks import ModelInterpretabilityCallback, LRCallback, DropoutCallback
from src.model_zoo import deep_cnn, shallow_cnn
from src.utils import _get_synthetic_data
from src.regularizers import SpectralNormRegularizer
from train_model_utils import get_keyboard_arguments
from train_model_utils import (models, BASERESULTSDIR, SYNTHETIC_DATADIR)

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras.callbacks import ModelCheckpoint
TASKDIR = "task_5_sub_6"

def main():
    # get keyboard arguments ; defined in the file train_model_utils.py
    args = get_keyboard_arguments()

    # get data
    traindata, validdata, testdata, model_test = _get_synthetic_data(SYNTHETIC_DATADIR)
    traindata = tf.data.Dataset.from_tensor_slices(traindata)
    validdata = tf.data.Dataset.from_tensor_slices(validdata)

    # set up the checkpoint directory
    BN = 'bn' if args.bn else 'no_bn'
    first_act = args.activation
    RESULTSDIR = os.path.join(BASERESULTSDIR, TASKDIR,f"{BN}", f"{first_act}")
    CKPTDIR = os.path.join(
                        RESULTSDIR,
                        f"{args.type.lower()}_{args.factor}",
                         )
    
    # do several trials
    for trial in range(args.start_trial, args.end_trial+1):

        # instantiate model
        get_model = models[args.type.lower()].get_model
        model = get_model(
                    input_shape=(200,4),
                    num_outputs=1,
                    bn=args.bn,
                    dropout1=0.1,
                    dropout2=0.5,
                    kernel_regularizer=SpectralNormRegularizer(1e-2, 20),
                    activation=first_act,
                    name="model",
                    factor=args.factor,
                    logits_only=False,
                        )
    
        # set up compile options and compile the model
        acc = tfk.metrics.BinaryAccuracy(name='acc')
        auroc = tfk.metrics.AUC(curve='ROC', name='auroc')
        aupr = tfk.metrics.AUC(curve='PR', name='aupr')
        optimizer = tfk.optimizers.Adam(learning_rate=1e-3)
        loss = tfk.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
        model.compile(optimizer=optimizer, loss=loss, metrics=[acc, auroc, aupr], )

        # set up callbacks
        callbacks = []
        if args.es:
            es_callback = tfk.callbacks.EarlyStopping(
                                            monitor='val_auroc',
                                            patience=10,
                                            verbose=1,
                                            mode='max',
                                            restore_best_weights=True
                                                    )
            callbacks.append(es_callback)
        if args.reduce_lr:
            reduce_lr = tfk.callbacks.ReduceLROnPlateau(
                                            monitor='val_auroc',
                                            factor=0.2,
                                            patience=4,
                                            min_lr=1e-7,
                                            mode='max',
                                            verbose=1
                                                )
            callbacks.append(reduce_lr)

        ## directory to save results from current trial
        ckptdir = os.path.join(CKPTDIR, f"trial_{trial:02d}")
        if not os.path.exists(ckptdir):
            os.makedirs(ckptdir)
        csvlogger = tfk.callbacks.CSVLogger(os.path.join(ckptdir, "log.csv"))
        ckpt_callback = tfk.callbacks.ModelCheckpoint(
                                        filepath=os.path.join(ckptdir, "ckpt_epoch-{epoch:04d}"),
                                        monitor="val_auroc",
                                        verbose=1,
                                        mode='max',
                                        save_best_only=True,
                                        save_weights_only=False,
                                        save_freq="epoch",
                                            )
        interp_callback = ModelInterpretabilityCallback(
                                            testdata=testdata,
                                            model_test=model_test,
                                            ckptdir=ckptdir,
                                            track_saliency=True,
                                            track_sg=True,
                                            track_intgrad=True,
                                                )
        dropout_callback = DropoutCallback(initial_rate=0., final_rate=0.4, wait=2)
        lr_callback = LRCallback(wait=2, factor=10.)
        callbacks = callbacks +\
            [csvlogger, ckpt_callback, interp_callback, dropout_callback, lr_callback]

        # fit the model
        model.fit(
            traindata.shuffle(10000).batch(32),
            validation_data=validdata.batch(64),
            epochs=100,
            callbacks=callbacks,
                )

if __name__ == '__main__':
    main()
        
        