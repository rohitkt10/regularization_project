import numpy as np, os, sys, h5py
import pandas as pd
from multiprocessing import Pool

import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
RESDIR = os.path.abspath("../results")

def plots_single_model(modeldir):
    """
    Generate all necessary plots for all trials from a single model directory.
    """
    # set up the directory to save plots
    plotsdir = os.path.join(modeldir,  'plots')
    if not os.path.exists(plotsdir):
        os.makedirs(plotsdir)

    # get all trials in the current model directory
    trialdirs = [os.path.join(modeldir, f) for f in os.listdir(modeldir) if 'trial' in f]

    # load the logs and interp data for all trials
    f_best_files = [h5py.File(os.path.join(trialdir, 'best_model_interp_scores.h5'),'r') \
                            for trialdir in trialdirs]
    f_all_files = [h5py.File(os.path.join(trialdir, 'all_epochs_interp_scores.h5'), 'r') \
                            for trialdir in trialdirs]
    log_files = [pd.read_csv(os.path.join(trialdir, 'log.csv'), index_col=0) \
                            for trialdir in trialdirs]

    # plot training dynamics
    val_acc = np.array([log['val_acc'].values for log in log_files])  ## (numtrail, numepoch)
    val_aupr = np.array([log['val_aupr'].values for log in log_files]) ## (numtrial, numepoch)
    val_auroc = np.array([log['val_auroc'].values for log in log_files]) ## (numtrial, numepoch)
    saliency_pr = np.array([list(map(lambda x : f[f"{x}/saliency/pr"][:].mean(0), np.arange(100))) for f in f_all_files])
    saliency_roc = np.array([list(map(lambda x : f[f"{x}/saliency/roc"][:].mean(0), np.arange(100))) for f in f_all_files])
    smoothgrad_pr = np.array([list(map(lambda x : f[f"{x}/smoothgrad/pr"][:].mean(0), np.arange(100))) for f in f_all_files])
    smoothgrad_roc = np.array([list(map(lambda x : f[f"{x}/smoothgrad/roc"][:].mean(0), np.arange(100))) for f in f_all_files])
    intgrad_pr = np.array([list(map(lambda x : f[f"{x}/intgrad/pr"][:].mean(0), np.arange(100))) for f in f_all_files])
    intgrad_roc = np.array([list(map(lambda x : f[f"{x}/intgrad/roc"][:].mean(0), np.arange(100))) for f in f_all_files])

    # plot training dynamics
    fig = plt.figure(figsize=(20, 15))

    ax = fig.add_subplot(221)
    ax.plot(epochs, val_acc.mean(0), linestyle='--', linewidth=3)
    ax.fill_between(epochs, val_acc.min(0), val_acc.max(0), label='Classification accuracy', alpha=0.5)
    ax.plot(epochs, val_aupr.mean(0), linewidth=3)
    ax.fill_between(epochs, val_aupr.min(0), val_aupr.max(0), label='Classification AUPR', alpha=0.5)
    ax.plot(epochs, val_auroc.mean(0), linewidth=3)
    ax.fill_between(epochs, val_auroc.min(0), val_auroc.max(0), label='Classification AUROC', alpha=0.5)
    ax.legend(loc='best', fontsize=14)
    ax.set_xlabel("Epoch", fontsize=15)

    ax = fig.add_subplot(222)
    ax.plot(epochs, saliency_pr.mean(0), linestyle='--', linewidth=3)
    ax.fill_between(epochs, saliency_pr.min(0), saliency_pr.max(0), label='Saliency AUPR', alpha=0.5)
    ax.plot(epochs, saliency_roc.mean(0), linestyle='--', linewidth=3)
    ax.fill_between(epochs, saliency_roc.min(0), saliency_roc.max(0), label='Saliency AUROC', alpha=0.5)
    ax.legend(loc='best', fontsize=14)
    ax.set_xlabel("Epoch", fontsize=15)

    ax = fig.add_subplot(223)
    ax.plot(epochs, smoothgrad_pr.mean(0), linestyle='--', linewidth=3)
    ax.fill_between(epochs, smoothgrad_pr.min(0), smoothgrad_pr.max(0), label='Smoothgrad AUPR', alpha=0.5)
    ax.plot(epochs, smoothgrad_roc.mean(0), linestyle='--', linewidth=3)
    ax.fill_between(epochs, smoothgrad_roc.min(0), smoothgrad_roc.max(0), label='Smoothgrad AUROC', alpha=0.5)
    ax.legend(loc='best', fontsize=14)
    ax.set_xlabel("Epoch", fontsize=15)

    ax = fig.add_subplot(224)
    ax.plot(epochs, intgrad_pr.mean(0), linestyle='--', linewidth=3)
    ax.fill_between(epochs, intgrad_pr.min(0), intgrad_pr.max(0), label='Integrated gradients AUPR', alpha=0.5)
    ax.plot(epochs, intgrad_roc.mean(0), linestyle='--', linewidth=3)
    ax.fill_between(epochs, intgrad_roc.min(0), intgrad_roc.max(0), label='Integrated gradients AUROC', alpha=0.5)
    ax.legend(loc='best', fontsize=14)
    ax.set_xlabel("Epoch", fontsize=15)


    fig.tight_layout()
    fig.savefig("performance_vs_interp_epochs.pdf", dpi=300)

    # plot training dynamics of interpretability metrics









def save_interpretability_plots(taskdir, model_type, factor):
    taskdir = os.path.join(RESDIR, taskdir)
    assert os.path.isdir(taskdir)

    # get the model type directory
    modeldirs = glob(os.path.join(taskdir, "**", f"{model_type}_{int(factor)}"), recursive=True)

    # execute parallelly for each model ; each thread processes all trials for a single model
    pool = Pool(len(modeldirs))
    pool.map(func=plots_single_trial, iterable=modeldirs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--subtask", type=int, default=1)
    parser.add_argument("--subsubtask", type=int, default=0)
    parser.add_argument("--type", type=str, default='deep')
    parser.add_argument("--factor", type=int, default=1)
    args = parser.parse_args()

    if not args.subsubtask:
        taskdir = f"task_{args.task}_sub_{args.subtask}"
    else:
        taskdir = f"task_{args.task}_sub_{args.subtask}_{args.subsubtask}"
    save_interpretability_plots(taskdir)
if __name__ == '__main__':
    main()
