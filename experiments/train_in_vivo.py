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
from src.utils import get_invivo_data
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

# train the model
def train(
        model, 
        ckptdir, 
        traindata, 
        validdata, 
        testdata, 
        model_test, 
        batchsize=128,
        num_epochs=100, 
        es_patience=10,
        adversarial=False, 
        method='smoothgrad',
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
    savepath = os.path.join(ckptdir, 'log.csv')

    # interp setup
    x_test, y_test = testdata
    threshold, top_k = 0.1, 10
    pos_index = np.where(y_test[:,0] == 1.)[0]
    num_analyze = len(pos_index)
    X = x_test[pos_index[:num_analyze]]
    X_model = model_test[pos_index[:num_analyze]]

    # compile model 
    model = compile_model(model, learning_rate=init_lr)

    # set up the entropy callback 
    x_valid, y_valid = validdata
    entropy_callback = EntropyModelSelectionCallback(x_valid, y_valid, method=method, output_layer_index=-1)
    entropy_callback.set_model(model)

    # set up results dictionary 
    all_results = setup_results_dict()
    
    # training loop 
    es_model = model 
    es_val_auroc = 0.
    wait = 0
    es = False
    best_entropy = np.inf
    entropy_callback.on_train_begin()
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
                traindata.shuffle(10000).batch(batchsize), 
                validation_data=validdata, 
                epochs=1,
                    )
        
        # update log file.
        entropy = entropy_callback._get_entropy()
        all_results['attr_entropy'].append(entropy)
        history = hist.history
        interp_res = calculate_interp_results(
                                        model=model, 
                                        X=X, 
                                        X_model=X_model, 
                                        threshold=threshold, 
                                        top_k=top_k, 
                                        saliency=saliency,
                                        smoothgrad=smoothgrad,
                                        intgrad=intgrad,
                                        use_snr=use_snr,
                                            )
        val_auroc = history['val_auroc'][0]
        all_results['train_auroc'].append(history['auroc'][0])
        all_results['valid_auroc'].append(history['val_auroc'][0])
        for k, v in interp_res.items():
            all_results[k].append(v)
        
        # save log data to disk
        df=pd.DataFrame(data=all_results, index=np.arange(epoch+1))
        df.index.name='epoch'
        df.to_csv(savepath)

        # check for entropy based model selection ; skip the first few epochs 
        if epoch >= 5:
            if entropy < best_entropy:
                best_entropy = entropy 
                model.save(os.path.join(ckptdir, f"entropy_es_model.h5"))
        
        # early stopping checks 
        if not es:
            current = val_auroc
            wait += 1
            if current > es_val_auroc:
                es_val_auroc = current
                wait = 0
                es_model = model
            if wait >= es_patience:
                es = True
                model.save(os.path.join(ckptdir, f'es_model.h5'))

    # save final model 
    model.save(os.path.join(ckptdir, 'final_model.h5'))

def main():
    return

if __name__=='__main__':
    main()