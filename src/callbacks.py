import numpy as np, h5py, pandas as pd,os
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import callbacks as tfkc
import tfomics
from tfomics import explain
from pdb import set_trace as keyboard

class ModelEvaluationCallback(tfk.callbacks.Callback):
    def __init__(self, x, y=None, batch_size=None, steps=None, filepath=None, save_freq=None):
        super().__init__()
        self.filepath = filepath
        if not y:
            assert isinstance(x,tf.data.Dataset), "Either pass tf.data.Dataset object or both x, y tensors."
        if isinstance(x, tf.data.Dataset):
            batch_size = None
        else:
            if not batch_size:
                batch_size = 32
        if not save_freq:
            self.save_freq = 1
        else:
            assert isinstance(save_freq, int)
            self.save_freq = save_freq
        self.options = {'x':x, 'y':y, 'batch_size':batch_size, 'return_dict':True, 'steps':steps}

    def on_epoch_end(self, epoch, logs=None):
        model = self.model
        #print(type(model))
        #print(model)
        res = model.evaluate(**self.options)
        printstr = "EVALUATION RESULTS: \n"
        for k in res.keys():
            printstr += f" - {k}: {res[k]}"
        print(printstr)

        # write results to disk
        if self.filepath:
            if not os.path.exists(os.path.dirname(self.filepath)):
                os.makedirs(os.path.dirname(self.filepath))
            df = pd.DataFrame(columns=res.keys(), data=np.array([list(res.values())]), index=[epoch+1])
            if not os.path.exists(self.filepath):
                print(f"Saving to location : {self.filepath}")
                df.to_csv(self.filepath,)
            else:
                _df = pd.read_csv(self.filepath, index_col=0)
                df = pd.concat([_df, df])
                df.to_csv(self.filepath,)

class ModelInterpretabilityCallback(tfkc.Callback):
    def __init__(
                self,
                testdata,
                model_test,
                ckptdir,
                num_analyze=None,
                save_freq=1,
                threshold=0.1,
                track_saliency=True,
                track_sg=True,
                track_intgrad=True,
                **kwargs
                ):
        """
        Parameters:
        ----------
        1. testdata <tuple> - A tuple consisting of test sequence-label data (x_test, y_test)
        2. model_test <numpy.ndarray> Ground truth
        3. ckptdir <str> - Directory where results are saved.
        4. num_analyze <int> - Number of examples to use for saliency analysis.
        5. save_freq <int> - Frequency (in epochs) with which to save interpretability results.

        """
        super().__init__(**kwargs)
        assert track_saliency or track_sg or track_intgrad, 'Atleast one type of attribution must be tracked.'
        self.ckptdir = ckptdir
        self.savedir = os.path.join(self.ckptdir, "attribution_scores",)
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        self.save_freq = save_freq
        self.track_saliency = track_saliency
        self.track_intgrad = track_intgrad
        self.track_sg = track_sg


        x_test, y_test = testdata
        pos_index = np.where(y_test[:,0] == 1.)[0]
        if not num_analyze:
            num_analyze = len(pos_index)
        self.X = x_test[pos_index[:num_analyze]]
        self.X_model = model_test[pos_index[:num_analyze]]
        self.threshold = threshold

    def _get_attribution_scores(self):
        print("Getting attribution scores...")
        scores = {}
        explainer = explain.Explainer(self.model, class_index=0)
        if self.track_saliency:
            scores['saliency'] = explainer.saliency_maps(self.X,)
        if self.track_sg:
            scores['smoothgrad'] = explainer.smoothgrad(self.X, num_samples=50, mean=0.0, stddev=0.1)
        if self.track_intgrad:
            scores['intgrad'] = explainer.integrated_grad(self.X, baseline_type='zeros')
        return scores

    def _save_data(self, data_dict, filepath,):
        """
        Save a dictionary of datasets in h5 format.
        """
        with h5py.File(filepath, "w") as file:
            #keyboard()
            for name, data in data_dict.items():
                file.create_dataset(name=name, data=data, compression='gzip')

    def on_train_begin(self, logs=None):
        print("Computing the attribution scores before training starts ...")
        scores = self._get_attribution_scores()
        filepath = os.path.join(self.savedir, "epoch_000.h5")
        self._save_data(scores, filepath)

    def on_epoch_end(self, epoch, logs=None):
        if epoch%self.save_freq == 0:
            scores = self._get_attribution_scores()
            filepath = os.path.join(self.savedir, f"epoch_{epoch:03d}.h5")
            self._save_data(scores, filepath)

class LRCallback(tfk.callbacks.Callback):
    def __init__(self, wait=5, factor=10.):
        self.wait = wait 
        self.factor = factor
    
    def on_epoch_end(self, epoch, logs=None):
        old_lr = tfk.backend.get_value(self.model.optimizer.lr)
        if self.wait-1 == epoch:
            tfk.backend.set_value(self.model.optimizer.lr, old_lr*self.factor)
            print(f"Learning rate increased to {tfk.backend.get_value(self.model.optimizer.lr)}")