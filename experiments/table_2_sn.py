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
from src.callbacks import BurnInCallback, CustomCSVLogger
from src.augmentations import GaussianNoiseAugmentation
from src.models import AugmentedModel, ManifoldGaussianNoiseModel, ModelWrapper
from src.regularizers import SpectralNormRegularizer

# parameters 
BATCHSIZE = 32
EPOCHS = 100
DROPOUT1 = 0.4
DROPOUT2 = 0.5
TABLEDIR = 'table_2'
NUMTRIALS = 5

def setup_results_dict():
    all_results = {}
    all_results['train_auroc'] = []
    all_results['valid_auroc'] = []
    all_results['test_auroc'] = []
    all_results['saliency_roc'] = []
    all_results['smoothgrad_roc'] = []
    all_results['intgrad_roc'] = []
    all_results['saliency_pr'] = []
    all_results['smoothgrad_pr'] = []
    all_results['intgrad_pr'] = []
    all_results['sal_snr'] = []
    all_results['sg_snr'] = []
    all_results['int_snr'] = []
    return all_results

# function to calculate sal, sg and intgrad scores 
def calculate_interp_results(model, X, X_model, threshold, top_k):
    res = {}
    explainer = explain.Explainer(model, class_index=0)
    
    saliency_scores = explainer.saliency_maps(X)
    sal_scores = explain.grad_times_input(X, saliency_scores)
    saliency_roc, saliency_pr = evaluate.interpretability_performance(sal_scores, X_model, threshold)
    sal_signal, sal_noise_max, sal_noise_mean, sal_noise_topk = evaluate.signal_noise_stats(sal_scores, X_model, top_k, threshold)
    sal_snr = evaluate.calculate_snr(sal_signal, sal_noise_topk) 
    res['saliency_roc']=np.mean(saliency_roc)
    res['saliency_pr']=np.mean(saliency_pr)
    res['sal_snr']=np.nanmean(sal_snr)
    
    
    smoothgrad_scores = explainer.smoothgrad(X, num_samples=50, mean=0.0, stddev=0.1)
    sg_scores = explain.grad_times_input(X, smoothgrad_scores)
    smoothgrad_roc, smoothgrad_pr = evaluate.interpretability_performance(sg_scores, X_model, threshold)
    sg_signal, sg_noise_max, sg_noise_mean, sg_noise_topk = evaluate.signal_noise_stats(sg_scores, X_model, top_k, threshold)
    sg_snr = evaluate.calculate_snr(sg_signal, sg_noise_topk)
    res['smoothgrad_roc']=np.mean(saliency_roc)
    res['smoothgrad_pr']=np.mean(saliency_pr)
    res['sg_snr']=np.nanmean(sal_snr)
    
    intgrad_scores = explainer.integrated_grad(X, baseline_type='zeros')
    int_scores = explain.grad_times_input(X, intgrad_scores)
    intgrad_roc, intgrad_pr = evaluate.interpretability_performance(int_scores, X_model, threshold)
    int_signal, int_noise_max, int_noise_mean, int_noise_topk = evaluate.signal_noise_stats(int_scores, X_model, top_k, threshold)
    int_snr = evaluate.calculate_snr(int_signal, int_noise_topk)
    res['intgrad_roc']=np.mean(saliency_roc)
    res['intgrad_pr']=np.mean(saliency_pr)
    res['int_snr']=np.nanmean(sal_snr)
    return res

# define the model
def deepnet(
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

# build the model given a learning rate
def compile_model(model, lr):
    acc = tfk.metrics.BinaryAccuracy(name='acc')
    auroc = tfk.metrics.AUC(curve='ROC', name='auroc')
    aupr = tfk.metrics.AUC(curve='PR', name='aupr')
    optimizer = tfk.optimizers.Adam(learning_rate=lr)
    loss = tfk.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
    model.compile(optimizer=optimizer, loss=loss, metrics=[acc, auroc, aupr], )
    return model

# train the model
def train(model, ckptdir, traindata, validdata, testdata, model_test, num_epochs=100):
    # savepath for log
    savepath = os.path.join(ckptdir, 'log.csv')
    
    # interp setup
    x_test, y_test = testdata
    threshold, top_k = 0.1, 10
    pos_index = np.where(y_test[:,0] == 1.)[0]
    num_analyze = len(pos_index)
    X = x_test[pos_index[:num_analyze]]
    X_model = model_test[pos_index[:num_analyze]]
    
    # train
    all_results = setup_results_dict()
    model = compile_model(model, 1e-3)
    model.training = False
    print("===============================================")
    print(f"Burn-in phase. Current learning rate = {tfk.backend.get_value(model.optimizer.learning_rate)}")
    print("===============================================")
    for i in range(2):
        print(f"Epoch {i+1}")
        print("-----------")
        model.fit(traindata.shuffle(10000).batch(32),validation_data=validdata.batch(64),epochs=1,)
        history = model.history.history
        all_results['train_auroc'].append(history['auroc'][0])
        all_results['valid_auroc'].append(history['val_auroc'][0])
        testres = model.evaluate(x_test, y_test, return_dict=True)
        all_results['test_auroc'].append(testres['auroc'])
        interp_res = calculate_interp_results(model, X, X_model, threshold, top_k)
        for k, v in interp_res.items():
            all_results[k].append(v)
        
        # save data to disk
        df=pd.DataFrame(data=all_results, index=np.arange(i+1))
        df.index.name='epoch'
        df.to_csv(savepath)
    
    # post burn in phase
    model = compile_model(model, 1e-2)
    model.training = True
    print("======================================================================")
    print(f"Burn-in phase complete. Turning on BN/DO.\nCurrent learning rate = {tfk.backend.get_value(model.optimizer.learning_rate)}")
    print("======================================================================")
    for i in range(2, num_epochs):
        print(f"Epoch {i+1}")
        print("-----------")
        model.fit(traindata.shuffle(10000).batch(32),validation_data=validdata.batch(64),epochs=1,)
        history = model.history.history
        all_results['train_auroc'].append(history['auroc'][0])
        all_results['valid_auroc'].append(history['val_auroc'][0])
        testres = model.evaluate(x_test, y_test, return_dict=True)
        all_results['test_auroc'].append(testres['auroc'])
        interp_res = calculate_interp_results(model, X, X_model, threshold, top_k)
        for k, v in interp_res.items():
            all_results[k].append(v)

        # save data to disk
        df=pd.DataFrame(data=all_results, index=np.arange(i+1))
        df.index.name='epoch'
        df.to_csv(savepath)
    return df
    

def main():
    # get keyboard arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", default='relu', help='First layer activation.', type=str)
    parser.add_argument("--nobn", action="store_true", help="Whether NOT to use batch norm",)
    parser.add_argument("--bn", action="store_true", help="Whether to use batch norm",)
    parser.add_argument("--factor", default=1, help='Factor.', type=int)
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
    #kernel_regularizer=None
    kernel_regularizer=SpectralNormRegularizer(lmbda=1e-3, n_iter=15)
    CKPTDIR = os.path.join(BASERESULTSDIR, TABLEDIR, BN, activation, 'sn')
    if not os.path.exists(CKPTDIR):
        os.makedirs(CKPTDIR)
    
    # do several trials
    for i in range(NUMTRIALS):
        print("*************")
        print(f"Trial : {i+1}")
        print("*************")
        factor = args.factor
        print(f"Model : deep - {factor}")
        print('----------------------')
        model = deepnet(
            input_shape=(200,4),
            num_outputs=1,
            bn=True,
            dropout1=0.4,
            dropout2=0.5,
            kernel_regularizer=kernel_regularizer,
            activation=args.activation,
            name="model",
            factor=args.factor,
            logits_only=False,
                )
        
        #augmentation = GaussianNoiseAugmentation(stddev=0.15)
        #model = AugmentedModel(model=model, augmentations=[augmentation])
        # model = ManifoldGaussianNoiseModel(model=model, stddev=0.15)
        model = ModelWrapper(model)
        
        # create ckpt directory if it doesnt exist
        ckptdir = os.path.join(CKPTDIR, f"deep_{factor}", f"trial_{i+1:02d}")
        if not os.path.exists(ckptdir):
            os.makedirs(ckptdir)
        
        # train
        train(model, ckptdir, traindata, validdata, testdata, model_test,)
            

if __name__=='__main__':
    main()