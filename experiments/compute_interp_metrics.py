import numpy as np, os, h5py, pandas as pd, sys, argparse
from multiprocessing import Pool
from glob import glob
from train_model_utils import SYNTHETIC_DATADIR
from pdb import set_trace as keyboard
import tensorflow as tf
from tensorflow import keras as tfk
import tfomics
from tfomics import impress, evaluate, explain
sys.path.append('..')
from src.utils import _get_synthetic_data
RESDIR = os.path.abspath("../results")

def interp_metrics(model, X, X_model, scores_file=None):
    """
    This function calculates the interpretability metrics given a model,
    test positive sequences, and ground truth positive sequences.
    The calculated values are returned as a dictionary.

    Parameters
    ----------
    1. model  - keras model to conduct interpretability analysis on.
    2. X - Test input sequences.
    3. X_model - Ground truth sequences.
    4. scores_file - h5 file containing precomputed attribution scores ; if not provided calculated from scratch.
    """
    threshold = 0.1
    top_k = 10

    if not scores_file:
        explainer = explain.Explainer(model, class_index=0)  ## set up explainer object
        saliency_scores = explainer.saliency_maps(X)
        smoothgrad_scores = explainer.smoothgrad(X, num_samples=50, mean=0.0, stddev=0.1)
        intgrad_scores = explainer.integrated_grad(X, baseline_type='zeros')
    else:
        scores_data = h5py.File(scores_file, 'r')
        saliency_scores = scores_data['saliency'][:]
        smoothgrad_scores = scores_data['smoothgrad'][:]
        intgrad_scores = scores_data['intgrad'][:]
        scores_data.close()
    scores_dict = {'saliency':saliency_scores, 'smoothgrad':smoothgrad_scores, 'intgrad':intgrad_scores}

    # compute interp metrics and save into a dictionary
    res = {}  ## dictionary of dictionaries
    for name, _scores in scores_dict.items():
        res[name] = {}

        scores = explain.grad_times_input(X, _scores)
        roc, pr = evaluate.interpretability_performance(scores, X_model, threshold)
        signal, noise_max, noise_mean, noise_topk = evaluate.signal_noise_stats(scores, X_model, top_k, threshold)
        snr = evaluate.calculate_snr(signal, noise_topk)

        res[name]['roc'] = roc
        res[name]['pr'] = pr
        res[name]['snr'] = snr #.create_dataset(name='snr', data=snr, compression='gzip')

    return res

# interpretability performance
def save_interpretability_results(taskdir):
    """
    Parse all available trial directories in the task directory.

    For each trial directory calculate the attribution ROC, PR and
    SNR for: 1. The best model checkpoint with the best classification performance,
    2. all states of the model after every training epoch.

    """
    # set up the task directory
    taskdir = os.path.join(RESDIR, taskdir)
    assert os.path.isdir(taskdir)

    ## load test data and ground truth data
    print("Loading data ...")
    _, _, (x_test, y_test), model_test = _get_synthetic_data(SYNTHETIC_DATADIR)
    pos_index = np.where(y_test[:,0] == 1)[0]
    num_analyze = len(pos_index)
    X = x_test[pos_index[:num_analyze]]
    X_model = model_test[pos_index[:num_analyze]]

    # get the log files
    logfiles = glob(os.path.join(taskdir, "**", "log.csv"), recursive=True)
    trialdirs = [f.split("log.csv")[0] for f in logfiles]

    #keyboard()

    ## loop over all trials
    for trialdir in trialdirs:
        print(f"Calculating results in directory: {trialdir}")
        attrdir = os.path.join(trialdir, 'attribution_scores') ## where attribution scores are stored

        # get the best checkpoint ; compute interp. metrics (ROC, PR, SNR)
        print("-----------------------------------------------")
        print("Calculating best model interpretability performance ...")
        checkpoints = [os.path.join(trialdir, f) for f in os.listdir(trialdir) if 'ckpt' in f]
        checkpoints.sort(key=lambda x : os.path.getmtime(x))
        best_ckpt = checkpoints[-1]
        model = tfk.models.load_model(best_ckpt)
        metrics = interp_metrics(
                            model=model,
                            X=X,
                            X_model=X_model,
                                )
        res = h5py.File(
                os.path.join(trialdir, 'best_model_interp_scores.h5'), 'w'
                 )
        for group, group_dict in metrics.items():
            res.create_group(group)
            for name, data in group_dict.items():
                res[group].create_dataset(name=name, data=data, compression='gzip')
        res.close()

        ## get all the attribution scores h5 filepaths
        attrscoresfiles = [os.path.join(attrdir, f) for f in np.sort(os.listdir(attrdir)) if 'epoch' in f]

        # compute the interp metrics on all trainign epochs
        print("------------------------------------")
        print("Calculating model interpretability at every training epoch ...")
        metrics = {}
        for i, scores_file in enumerate(attrscoresfiles):
            print(f"Currently on epoch : {i}")
            metrics[i] = interp_metrics(model, X, X_model, scores_file=scores_file)
        print("------------------------------------")

        # save results
        res = h5py.File(
                os.path.join(trialdir, 'all_epochs_interp_scores.h5'), 'w'
                    )
        for epoch, epoch_metrics in metrics.items():
            res.create_group(str(epoch))
            for group, group_dict in epoch_metrics.items():
                res[str(epoch)].create_group(group)
                for name, data in group_dict.items():
                    res[str(epoch)][group].create_dataset(name=name, data=data, compression='gzip')
        res.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--subtask", type=int, default=1)
    parser.add_argument("--subsubtask", type=int, default=0)
    # parser.add_argument("--type", type=str, default='deep')
    # parser.add_argument("--factor", type=int, default=1)
    # parser.add_argument("--trial", type=int, default=1)
    args = parser.parse_args()
    #assert args.type in ['deep', 'shallow']
    #assert args.factor in [1, 4, 8]

    # set up the directory for the requested task and subtask
    if not args.subsubtask:
        taskdir = f"task_{args.task}_sub_{args.subtask}"
    else:
        taskdir = f"task_{args.task}_sub_{args.subtask}_{args.subsubtask}"
    #taskdir = os.path.join(taskdir, f"{args.type}_{args.factor}", f"trial_{trial:02d})

    # save interpretability results for all trial directories in this task directory
    save_interpretability_results(taskdir)


if __name__ == '__main__':
    main()
