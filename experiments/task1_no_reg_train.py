"""
Task 1 - Comparing the classification performance and interpretability of
shallow and deep CNN models under various regularization settings.
-------------------------------------------------------------------

Subtask 1.1 - No regularization.

In this subtask we train the shallow and deep CNN without any regularization or
dropout. The test is conducted for 3 versions of the shallow and deep
CNNs each - scaling each of these types of models by a factor of 1, 4 and 8.

"""

import numpy as np, os, sys, h5py, pandas as pd, argparse
from pdb import set_trace as keyboard

sys.path.append("..")
from src.callbacks import ModelInterpretabilityCallback
from src.model_zoo import deep_cnn, shallow_cnn

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras.callbacks import ModelCheckpoint

# set up directories
RESULTSDIR = os.path.abspath("../results/task_1_sub_1")
DATADIR = os.path.abspath("../data")

# dictionary mapping names to model types
models = {'shallow':shallow_cnn, 'deep':deep_cnn}

# allowed scaling factors
factors = [1, 4, 8]

def _validate_args(args):
    assert int(args.factor) in factors
    assert args.type.lower() in models.keys()
    assert isinstance(args.batch, int) and args.batch > 0
    assert isinstance(args.epochs, int) and args.epochs > 0
    assert isinstance(args.numtrials, int) and args.numtrials > 0

def _get_synthetic_data(datadir):
    """
    This function expects to find a file named `synthetic_code_dataset.h5`
    in the given data directory.
    """
    filepath = os.path.join(DATADIR, 'synthetic_code_dataset.h5')
    with h5py.File(filepath, 'r') as dataset:
        x_train = np.array(dataset['X_train']).astype(np.float32)
        y_train = np.array(dataset['Y_train']).astype(np.float32)
        x_valid = np.array(dataset['X_valid']).astype(np.float32)
        y_valid = np.array(dataset['Y_valid']).astype(np.int32)
        x_test = np.array(dataset['X_test']).astype(np.float32)
        y_test = np.array(dataset['Y_test']).astype(np.int32)
        model_test = np.array(dataset['model_test']).astype(np.float32)
    model_test = model_test.transpose([0,2,1])
    x_train = x_train.transpose([0,2,1])
    x_valid = x_valid.transpose([0,2,1])
    x_test = x_test.transpose([0,2,1])
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), model_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=32, help="Batch size", type=int)
    parser.add_argument("--epochs", default=100, help="Epochs", type=int)
    parser.add_argument("--factor", default=1, help='Expansion factor', type=int)
    parser.add_argument("--type", default='deep', help='Deep or shallow model', type=str)
    parser.add_argument("--numtrials", default=10, help='Number of trials of each model run', type=int)
    args = parser.parse_args()
    _validate_args(args)

    # get data
    traindata, validdata, testdata, model_test = _get_synthetic_data(DATADIR)

    # set up the checkpoint directory
    ckptdir = os.path.join(
                        RESULTSDIR,
                        f"{args.type.lower()}_{args.factor}",
                         )
    ## get_model corresponding to the selected model type
    get_model = models[args.type.lower()].get_model

    # do 10 trials
    for trial in range(args.numtrials):
        ## directory to save results from current trial
        ckptdir = os.path.join(ckptdir, f"trial_{trial:02d}")
        if not os.path.exists(ckptdir):
            os.makedirs(ckptdir)

        # instantiate model
        model = get_model(
                    input_shape=(200, 4),
                    num_outputs=1,
                    bn=True,
                    factor=int(args.factor),
                    logits_only=False
                        )

        # set up compile options and compile the model
        acc = tfk.metrics.BinaryAccuracy(name='acc')
        auroc = tfk.metrics.AUC(curve='ROC', name='auroc')
        aupr = tfk.metrics.AUC(curve='PR', name='aupr')
        optimizer = tfk.optimizers.Adam(learning_rate=0.001)
        loss = tfk.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
        model.compile(optimizer=optimizer, loss=loss, metrics=[acc, auroc, aupr], )

        # set up callbacks
        es_callback = tfk.callbacks.EarlyStopping(
                                        monitor='val_auroc',
                                        patience=10,
                                        verbose=1,
                                        mode='max',
                                        restore_best_weights=True
                                                )
        reduce_lr = tfk.callbacks.ReduceLROnPlateau(
                                        monitor='val_auroc',
                                        factor=0.2,
                                        patience=4,
                                        min_lr=1e-7,
                                        mode='max',
                                        verbose=1
                                            )
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
                                                )
        callbacks = [es_callback, reduce_lr, ckpt_callback, interp_callback]

        # fit the model
        epochs = args.epochs
        batchsize = args.batch
        model.fit(
            traindata.batch(batchsize),
            validation_data=validdata.batch(64),
            epochs=epochs,
            callbacks=callbacks,
                )
    keyboard()

if __name__ == '__main__':
    main()
