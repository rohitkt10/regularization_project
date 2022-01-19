import numpy as np, h5py, pandas as pd,os
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import callbacks as tfkc
import tfomics
from tfomics import explain, evaluate
from pdb import set_trace as keyboard

# callback to log everything - train/valid/test classification and interpretabilty
class CustomCSVLogger(tfkc.Callback):
    """
    log the classification performance and interpretability after every epoch.
    """
    def setup_results_dict(self):
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
        self.all_results = all_results
    
    def __init__(self,filepath, testdata,model_test,threshold=0.1,**kwargs):
        """
        Parameters:
        ----------
        1. testdata <tuple> - A tuple consisting of test sequence-label data (x_test, y_test)
        2. model_test <numpy.ndarray> Ground truth
        3. ckptdir <str> - Directory where results are saved.

        """
        super().__init__(**kwargs)
        x_test, y_test = testdata
        self.x_test = x_test
        self.y_test = y_test
        self.filepath = filepath
        
        # get sequences to perform interp. analysis
        pos_index = np.where(self.y_test[:,0] == 1.)[0]
        num_analyze = len(pos_index)
        self.X = self.x_test[pos_index[:num_analyze]]
        self.X_model = model_test[pos_index[:num_analyze]]
        self.threshold = threshold
        self.setup_results_dict()
    
    def on_epoch_end(self, epoch, logs=None):
        # log classification performance (train, valid, test)
        testres = self.model.evaluate(self.x_test, self.y_test, return_dict=True)['auroc']
        #for key in list(self.all_results.keys())[3:]:
        #    self.all_results[key].append(-1)
        self.all_results['train_auroc'].append(logs['auroc'])
        self.all_results['valid_auroc'].append(logs['val_auroc'])
        self.all_results['test_auroc'].append(testres)
        
        # log interp. performance  (saliency, smoothgrad, intgrad - roc, pr, snr)
        explainer = explain.Explainer(self.model, class_index=0)
        saliency_scores = explainer.saliency_maps(self.X)
        smoothgrad_scores = explainer.smoothgrad(self.X, num_samples=50, mean=0.0, stddev=0.1)
        intgrad_scores = explainer.integrated_grad(self.X, baseline_type='zeros')
        sal_scores = explain.grad_times_input(self.X, saliency_scores)
        sg_scores = explain.grad_times_input(self.X, smoothgrad_scores)
        int_scores = explain.grad_times_input(self.X, intgrad_scores)
        saliency_roc, saliency_pr = evaluate.interpretability_performance(sal_scores, self.X_model, self.threshold)
        smoothgrad_roc, smoothgrad_pr = evaluate.interpretability_performance(sg_scores, self.X_model, self.threshold)
        intgrad_roc, intgrad_pr = evaluate.interpretability_performance(int_scores, self.X_model, self.threshold)
        top_k = 10
        sal_signal, sal_noise_max, sal_noise_mean, sal_noise_topk = evaluate.signal_noise_stats(sal_scores, self.X_model, top_k, self.threshold)
        sg_signal, sg_noise_max, sg_noise_mean, sg_noise_topk = evaluate.signal_noise_stats(sg_scores, self.X_model, top_k, self.threshold)
        int_signal, int_noise_max, int_noise_mean, int_noise_topk = evaluate.signal_noise_stats(int_scores, self.X_model, top_k, self.threshold)
        sal_snr = evaluate.calculate_snr(sal_signal, sal_noise_topk) 
        int_snr = evaluate.calculate_snr(int_signal, int_noise_topk)
        sg_snr = evaluate.calculate_snr(sg_signal, sg_noise_topk)
        self.all_results['saliency_roc'].append(np.mean(saliency_roc))
        self.all_results['smoothgrad_roc'].append(np.mean(smoothgrad_roc))
        self.all_results['intgrad_roc'].append(np.mean(intgrad_roc))
        self.all_results['saliency_pr'].append(np.mean(saliency_pr))
        self.all_results['smoothgrad_pr'].append(np.mean(smoothgrad_pr))
        self.all_results['intgrad_pr'].append(np.mean(intgrad_pr))
        self.all_results['sal_snr'].append(np.nanmean(sal_snr))
        self.all_results['sg_snr'].append(np.nanmean(sg_snr))
        self.all_results['int_snr'].append(np.nanmean(int_snr))
        
        # save df to dict 
        index = np.arange(epoch+1)
        df = pd.DataFrame(data=self.all_results, index=np.arange(epoch+1))
        df.index.name = 'epoch'
        df.to_csv(self.filepath)

# callback to implement the burn in
class BurnInCallback(tfkc.Callback):
    def __init__(self, wait=2, final_lr=1e-2):
        self.wait = wait
        self.final_lr = final_lr 
        super().__init__()
    
    def on_train_begin(self, logs=None):
        print("\nBeginning training. Dropout and batch norm switched off in the burn in phase")
        self.model.training = False
    
    def on_epoch_end(self, epoch, logs=None):
        if self.wait - 1 == epoch:
            tfk.backend.set_value(self.model.optimizer.learning_rate, self.final_lr)
            self.model.training = True
            print("\nBurn in phase over. Dropout and batch norm switched back on and learning rate increased.")



class ExtendedEarlyStopping(tfkc.EarlyStopping):
    """
    Early stopping + evaluate interpretability performance at early stopping. 
    """
    def __init__(
            self,
            max_epochs,
            testdata, 
            model_test,
            ckptdir,
            threshold=0.1,
            top_k=10, 
            *es_args, **es_kwargs
                 ):
        """
        Parameters
        ----------
         *es_args, **es_kwargs - arguments and keyword arguments passed to keras.callbacks.EarlyStopping
        """
        super().__init__(**es_kwargs)
        self.max_epochs = max_epochs
        self.ckptdir = ckptdir
        x_test, y_test = testdata
        pos_index = np.where(y_test[:,0] == 1.)[0]
        num_analyze = len(pos_index)
        self.X = x_test[pos_index[:num_analyze]]
        self.X_model = model_test[pos_index[:num_analyze]]
        self.threshold = threshold
        self.top_k = top_k
        self.es_interp = {}
    
    def _calculate_interptability(self):
        columns = ['Method', 'ROC mean', 'ROC std', 'PR mean', 'PR std', 'SNR mean', 'SNR std']
        data = []

        print("\nEarly stopping triggered. Calculating the interpretability scores...")
        explainer = explain.Explainer(self.model, class_index=0)

        # saliency maps 
        scores = explain.grad_times_input(self.X, explainer.saliency_maps(self.X,))
        roc, pr = evaluate.interpretability_performance(scores, self.X_model, self.threshold)
        signal, noise_max, noise_mean, noise_topk = evaluate.signal_noise_stats(scores, self.X_model, self.top_k, self.threshold)
        snr = evaluate.calculate_snr(signal, noise_topk)
        roc_mean, roc_std = np.mean(roc), np.std(roc)
        pr_mean, pr_std = np.mean(pr), np.std(pr)
        snr_mean, snr_std = np.nanmean(snr), np.nanstd(snr)
        _data = ['saliency', roc_mean, roc_std, pr_mean, pr_std, snr_mean, snr_std]
        data.append(_data)

        # smoothgrad 
        scores = explain.grad_times_input(self.X, explainer.smoothgrad(self.X, num_samples=50, mean=0.0, stddev=0.1))
        roc, pr = evaluate.interpretability_performance(scores, self.X_model, self.threshold)
        signal, noise_max, noise_mean, noise_topk = evaluate.signal_noise_stats(scores, self.X_model, self.top_k, self.threshold)
        snr = evaluate.calculate_snr(signal, noise_topk)
        roc_mean, roc_std = np.mean(roc), np.std(roc)
        pr_mean, pr_std = np.mean(pr), np.std(pr)
        snr_mean, snr_std = np.nanmean(snr), np.nanstd(snr)
        _data = ['smoothgrad', roc_mean, roc_std, pr_mean, pr_std, snr_mean, snr_std]
        data.append(_data)
        
        # intgrad 
        scores = explain.grad_times_input(self.X, explainer.integrated_grad(self.X, baseline_type='zeros'))
        roc, pr = evaluate.interpretability_performance(scores, self.X_model, self.threshold)
        signal, noise_max, noise_mean, noise_topk = evaluate.signal_noise_stats(scores, self.X_model, self.top_k, self.threshold)
        snr = evaluate.calculate_snr(signal, noise_topk)
        roc_mean, roc_std = np.mean(roc), np.std(roc)
        pr_mean, pr_std = np.mean(pr), np.std(pr)
        snr_mean, snr_std = np.nanmean(snr), np.nanstd(snr)
        _data = ['intgrad', roc_mean, roc_std, pr_mean, pr_std, snr_mean, snr_std]
        data.append(_data)

        # write to a csv file
        df = pd.DataFrame(columns=columns, data=data)
        df.index += 1
        df.to_csv(os.path.join(self.ckptdir, 'es_interp.csv'))

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch=epoch, logs=logs)
        if self.model.stop_training or (epoch == self.max_epochs - 1):
            # save the early stopping interp. score
            self._calculate_interptability()

            # save the early stopping index 
            es_epoch = epoch - self.wait
            fname = os.path.join(os.path.abspath(self.ckptdir), "es.txt")
            f = open(fname, "w")
            f.write(str(es_epoch+1))
            f.close()
            
class EarlyStoppingMarker(tfkc.EarlyStopping):
    """
    like `keras.callbacks.EarlyStopping` except training doesnt actually stop, we just 
    make a note of the epoch at which early stopping is triggered. 
    """
    def __init__(self, ckptdir=".", *args, **kwargs):
        """
        Parameters
        ----------
        1. ckptdir - Where to save the early stopping epoch number. 
        2. *args, **kwargs - arguments and keyword arguments passed to keras.callbacks.EarlyStopping
        """
        self.ckptdir = ckptdir
        self.es_epoch = None
        super().__init__(*args, **kwargs)
    
    def on_epoch_end(self, epoch, logs=None):
        if self.es_epoch == None:
            current = self.get_monitor_value(logs)
            if current is None:
                return
            
            if self.restore_best_weights and self.best_weights is None:
                # Restore the weights after first epoch if no progress is ever made.
                self.best_weights = self.model.get_weights()

            self.wait += 1
            if self._is_improvement(current, self.best):
                self.best = current
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
            
                # Only restart wait if we beat both the baseline and our previous best.
                if self.baseline is None or self._is_improvement(current, self.baseline):
                    self.wait = 0

            if self.wait >= self.patience:
                self.stopped_epoch = epoch ## counting from 0 
                self.es_epoch = self.stopped_epoch - self.wait
                print(f"\nEarly stopping epoch : {self.es_epoch+1}")

                ## dont stop training ; dont restore wait ; just log the ES index and continue training 
                fname = os.path.join(os.path.abspath(self.ckptdir), "es.txt")
                f = open(fname, "w")
                f.write(str(self.es_epoch+1))
                f.close()

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

class DropoutCallback(tfk.callbacks.Callback):
    def __init__(self, initial_rate, final_rate, wait=2):
        super().__init__()
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.wait = wait
    
    def _set_dropout_idx(self):
        if isinstance(self.model.layers[4], tfk.layers.Dropout):
            self.idx = 4
        else:
            self.idx = 3
    
    def on_train_begin(self, logs=None):
        self._set_dropout_idx()
        logs = logs or {}
        self.model.layers[self.idx].rate = self.initial_rate

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['dropout_1'] = self.model.layers[self.idx].rate
        if self.wait - 1 == epoch:
            self.model.layers[self.idx].rate = self.final_rate

class LRCallback(tfk.callbacks.Callback):
    def __init__(self, wait=5, factor=10.):
        self.wait = wait 
        self.factor = factor
    
    def on_epoch_end(self, epoch, logs=None):
        old_lr = tfk.backend.get_value(self.model.optimizer.lr)
        if self.wait-1 == epoch:
            tfk.backend.set_value(self.model.optimizer.lr, old_lr*self.factor)
            print(f"Learning rate increased to {tfk.backend.get_value(self.model.optimizer.lr)}")