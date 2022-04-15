"""
Model training pipeline.
-----------------------
Given a model type, run several trials of the model and generate attribution maps for all. 
"""
import numpy as np, os, h5py, sys, pandas as pd, argparse
from pdb import set_trace as keyboard

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
from tensorflow.keras import callbacks as tfkc
from tensorflow_probability import distributions as tfd
import tfomics
from tfomics import impress, explain, evaluate

sys.path.append("..")
from src.utils import get_invivo_data, IN_VIVO_NAMES
from src.attackers import PGDAttack, FGSMAttack
from src.regularizers import SpectralNormRegularizer
custom_objects={
        'SpectralNormRegularizer':SpectralNormRegularizer,
        'PGDAttack':PGDAttack, 
        'FGSMAttack':FGSMAttack,
        }

# -----------------------------------------------------------
# Set up parameters, directories, lists etc. 
REGULARIZERS = [
    'standard',
    'mixup',
    'manifold-mixup',
    'gn',
    'manifold-gn',
    'sn',
    'adversarial',
]
ATTR_METHODS = ['saliency', 'smoothgrad', 'intgrad']

# ----------------------------------------------------------

FILTERS = [196, 256, 512]

def get_model(
            input_shape, 
            num_outputs,
            bn=False,
            dropout1=0.2,
            dropout2=0.5,
            kernel_regularizer=None,
            activation='relu',
            name="model",
                ):
    # input layer
    inputs = tfk.layers.Input(shape=input_shape)

    # layer 1 - convolution
    nn = tfk.layers.Conv1D(filters=FILTERS[0], 
                            kernel_size=19, 
                            padding='same', 
                            kernel_regularizer=kernel_regularizer)(inputs) 
    if bn:       
        nn = tfk.layers.BatchNormalization()(nn)
    nn = tfk.layers.Activation(activation)(nn)
    nn = tfk.layers.MaxPool1D(pool_size=4)(nn)
    nn = tfk.layers.Dropout(dropout1)(nn)

    # layer 2 - convolution
    nn = tfk.layers.Conv1D(filters=FILTERS[1], 
                            kernel_size=9, 
                            padding='same', 
                            kernel_regularizer=kernel_regularizer)(nn)        
    if bn:
        nn = tfk.layers.BatchNormalization()(nn)
    nn = tfk.layers.Activation('relu')(nn)
    nn = tfk.layers.MaxPool1D(pool_size=4)(nn)
    nn = tfk.layers.Dropout(dropout1)(nn)

    # layer 3 - convolution
    nn = tfk.layers.Conv1D(filters=FILTERS[2], 
                            kernel_size=7, 
                            padding='same', 
                            kernel_regularizer=kernel_regularizer)(nn)
    if bn:     
        nn = tfk.layers.BatchNormalization()(nn)
    nn = tfk.layers.Activation('relu')(nn)
    nn = tfk.layers.MaxPool1D(pool_size=4)(nn)
    nn = tfk.layers.Dropout(dropout1)(nn)

    # layer 3 - Fully-connected 
    nn = tfk.layers.Flatten()(nn)
    nn = tfk.layers.Dense(1000)(nn)
    if bn:
        nn = tfk.layers.BatchNormalization()(nn)
    nn = tfk.layers.Activation('relu')(nn)
    nn = tfk.layers.Dropout(dropout2)(nn)

    # Output layer
    logits = tfk.layers.Dense(num_outputs, 
                                use_bias=True, 
                                kernel_regularizer=kernel_regularizer)(nn)
    outputs = tfk.layers.Activation('sigmoid')(logits)

    # create keras model
    model = tfk.Model(inputs=inputs, outputs=outputs)

    return model

def compile_model(
                model,
                optimizer=tfk.optimizers.Adam,
                learning_rate=1e-3,
                lossfn=tfk.losses.BinaryCrossentropy,
                accuracy=True,
                auroc=True,
                aupr=True,
                extra_metrics=[],
                opt_kwargs={},
                loss_kwargs={},
                ):
    """
    Compile a given model. The only required arg is the model
    object. If no other arguments are provided, then the model 
    is compiled with the following default choices:
    1. adam optimizer with learning rate 1e-3 and default moving average params.
    2. binary cross entropy loss applied to sigmoid outputs.
    3. accuracy, auroc, aupr metrics. 
    
    """
    # set up optimizer 
    opt = optimizer(learning_rate=learning_rate, **opt_kwargs)

    # set up loss function 
    loss = lossfn(**loss_kwargs)

    # set up metrics 
    metrics = []
    if accuracy:
        metrics.append(tfk.metrics.BinaryAccuracy(name='acc'))
    if auroc:
        metrics.append(tfk.metrics.AUC(curve='ROC', name='auroc'))
    if aupr:
        metrics.append(tfk.metrics.AUC(curve='PR', name='aupr'))
    metrics = metrics + extra_metrics
    if len(metrics) == 0:
        metrics = None 
    
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    return model

# train the model
def train(
        model, 
        ckptdir, 
        traindata, 
        validdata, 
        testdata, 
        batchsize=128,
        num_epochs=70, 
        es_monitor='aupr',
        es_patience=10,
        adversarial=False, 
        clean_epochs=5, 
        mix_epochs=20,
        es_start_epoch=35,
        warmup=False,
        warmup_epochs=2,
        init_lr=1e-3,
        final_lr=1e-2,
        ): 
    if adversarial:
        if clean_epochs > 0:
            model.clean = True
        else:
            model.clean = False
        model.mix = True

    # savepath for log
    logpath = os.path.join(ckptdir, 'log.csv')

    # compile model 
    model = compile_model(model, learning_rate=init_lr)
    
    # training loop 
    es_model = model 
    es_val_metric = 0.
    wait = 0
    es = False
    resdict = {}
    for epoch in range(num_epochs):
        print(f"... Epoch {epoch+1} ...")
        
        # check for warmup regularization modes.
        if warmup == True and epoch == 0:
            print("===============================================")
            print(f"Warmup phase.\
             Current learning rate = {tfk.backend.get_value(model.optimizer.learning_rate)}")
            print("===============================================")
            model.training = False 
        if warmup == True and epoch == warmup_epochs:
            model = compile_model(model, learning_rate=final_lr)
            model.training = True
            print("======================================================================")
            print(f"Burn-in phase complete. Turning on BN/DO.\nCurrent learning rate = {tfk.backend.get_value(model.optimizer.learning_rate)}")
            print("======================================================================")
        
        # check for adversarial training modes.
        if adversarial == True and epoch == clean_epochs: # switch on adversarial training
            print("Clean epochs done. Beginning adversarial mode.")
            model.clean = False
        if adversarial == True and epoch == mix_epochs: # switch off adversarial data augmentation. 
            print("Mixed adversarial mode done. Beginning pure adversarial mode.")
            model.mix = False
        
        # do 1 epoch of training.
        hist = model.fit(
                    traindata, 
                    validation_data=validdata, 
                    epochs=1,
                    )
        
        # update log file.
        history = hist.history
        if len(resdict) == 0:
            resdict = history 
        else:
            for key, value in history.items():
                resdict[key].append(value[0])
        df=pd.DataFrame(data=resdict, index=np.arange(epoch+1))
        df.index.name='epoch'
        df.to_csv(logpath)
        
        # early stopping checks 
        if not es:
            current = history[f'val_{es_monitor}'][0]
            wait += 1
            if current > es_val_metric:
                es_val_metric = current
                wait = 0
                es_model = model
            if wait >= es_patience:
                es = True
                model.save(os.path.join(ckptdir, f'es_model.h5'))

    # evaluate and save final model 
    testresdict = model.evaluate(testdata, return_dict=True)
    data = np.array(list(testresdict.values()))
    columns = list(testresdict.keys())
    df = pd.DataFrame(data=data[None, :], columns=columns)
    df.to_csv(os.path.join(ckptdir, 'testres.csv'))
    model.save(os.path.join(ckptdir, 'final_model.h5'))

def main():
    # get keyboard arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--resdir", default="../results/in_vivo", type=str, 
                        help='The base directory in which to store results.')
    parser.add_argument("--debug", action="store_true", 
                        help="whether to work in debug mode.")
    parser.add_argument("--datadir", default="../data/in_vivo", type=str,
                        help='The directory containing data.')
    parser.add_argument("--name", default="A549", type=str,
                        help=f'Name of the invivo dataset. Choices = {IN_VIVO_NAMES}')
    parser.add_argument("--epochs", default=70, 
            help='Number of epochs.', type=int)
    parser.add_argument("--batch", default=128, 
            help='Batchsize.', type=int)
    parser.add_argument("--dropout1", default=0.1, 
            help='Dropout after conv layers.', type=float)
    parser.add_argument("--dropout2", default=0.5, 
            help='Dropout after dense layers.', type=float)
    parser.add_argument("--activation", default='relu', 
            help='First layer activation.', type=str)
    parser.add_argument("--nobn", action="store_true", 
            help="Whether NOT to use batch norm",)
    parser.add_argument("--bn", action="store_true", 
            help="Whether to use batch norm",)
    parser.add_argument('--regularizer', type=str, default='standard', 
            help=f'What type of regularizer to use. Options : {REGULARIZERS}')
    parser.add_argument("--warmup", action="store_true", 
            help="Whether to use the warmup schedule.",)
    parser.add_argument("--warmup_epochs", type=int, default=2,
            help="Number of warmup epochs.")
    parser.add_argument("--init_lr", type=float, default=1e-3, 
            help='Initial learning rate. For the warm up schedule, this learning rate \
                    is followed in the burn-in phase. \
                    For the standard schedule, this learning rate is followed \
                    all the way through training.')
    parser.add_argument("--final_lr", type=float, default=1e-2,
            help='Final learning. For the warmup schedule, this learning rate \
                is followed in the post burn-in phase. For the standard schedule \
                this learning rate is not used.')
    args = parser.parse_args()

    # collect arguments 
    baseresultsdir = args.resdir 
    datadir = args.datadir
    dataset_name = args.name
    num_epochs = args.epochs 
    batchsize = args.batch
    activation = args.activation
    dropout1 = args.dropout1 
    dropout2 = args.dropout2
    bn = args.bn
    regularizer = args.regularizer
    assert regularizer in REGULARIZERS, \
        f'Invalid regularization choice. Available choices : {REGULARIZERS}'
    if regularizer == 'adversarial':
        adversarial = True
    else:
        adversarial = False
    warmup = args.warmup 
    warmup_epochs = args.warmup_epochs
    init_lr = args.init_lr 
    final_lr = args.final_lr
    
    # load data 
    traindata, validdata, testdata = get_invivo_data(datadir, name=dataset_name)
    _, L, A = traindata[0].shape
    num_outputs = traindata[1].shape[1]
    traindata = tf.data.Dataset.from_tensor_slices(traindata).shuffle(10000).batch(batchsize)
    validdata = tf.data.Dataset.from_tensor_slices(validdata).batch(64)
    testdata = tf.data.Dataset.from_tensor_slices(testdata).batch(64)
    
    # set up kernel regularizer
    kernel_regularizer=None
    if regularizer == 'sn':
        kernel_regularizer = SpectralNormRegularizer(1e-2, 15)
    
    # set up model
    model = get_model(
                    input_shape=(L, A),
                    num_outputs=num_outputs,
                    bn=bn,
                    dropout1=dropout1,
                    dropout2=dropout2,
                    activation=activation, 
                    kernel_regularizer=kernel_regularizer,
                        )
    if regularizer == 'gn':
        from src.augmentations import GaussianNoiseAugmentation
        from src.models import AugmentedModel
        augmentation = GaussianNoiseAugmentation(stddev=0.15)
        model = AugmentedModel(model=model, augmentations=[augmentation])
    elif regularizer == 'manifold-gn':
        from src.models import ManifoldGaussianNoiseModel
        model = ManifoldGaussianNoiseModel(model, stddev=0.15)
    elif regularizer == 'mixup':
        from src.augmentations import MixupAugmentation
        from src.models import AugmentedModel 
        augmentation = MixupAugmentation(alpha=1.0)
        model = AugmentedModel(model=model, augmentations=[augmentation])
    elif regularizer == 'manifold-mixup':
        from src.models import ManifoldMixupModel
        model = ManifoldMixupModel(model, alpha=1.0)
    elif regularizer == 'adversarial':
        from src.models import AdversariallyTrainedModel
        model = AdversariallyTrainedModel(model)
    else:
        from src.models import ModelWrapper
        model = ModelWrapper(model)

    # set up the checkpoint directory
    if bn:
        BN = 'bn'
    else:
        BN = 'nobn'
    if warmup:
        SCHEDULE = 'warmup'
    else:
        SCHEDULE = 'fixed_lr'
    ckptdir = os.path.join(
                    baseresultsdir,
                    dataset_name,
                    SCHEDULE, 
                    BN, 
                    activation, 
                    regularizer, 
                        )
    if not os.path.exists(ckptdir):
        trial = 1
    else:
        trials_done = len([x for x in os.listdir(ckptdir) if 'trial' in x])
        trial = trials_done + 1
    ckptdir = os.path.join(ckptdir, f'trial_{trial}')
    os.makedirs(ckptdir)

    # train model
    train(
        model=model, 
        ckptdir=ckptdir, 
        traindata=traindata, 
        validdata=validdata, 
        testdata=testdata, 
        batchsize=batchsize,
        num_epochs=num_epochs, 
        es_monitor='aupr',
        es_patience=10,
        adversarial=adversarial, 
        clean_epochs=5, 
        mix_epochs=20,
        es_start_epoch=35,
        warmup=warmup,
        warmup_epochs=warmup_epochs,
        init_lr=init_lr,
        final_lr=final_lr,
        )

if __name__=='__main__':
    main()