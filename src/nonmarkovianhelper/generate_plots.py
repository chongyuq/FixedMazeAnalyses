import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from scipy.stats import ttest_1samp, permutation_test
from pathlib import Path
from regressionhelper.constants import *
from datahelper.fields import COMMON_FIELDS
from regressionhelper.regression_loglikelihood_funcs import neg_log_likelihood
from collections import defaultdict


def plot_nonmarkovianness(maze_number=None, output_dir=None):
    assert maze_number in [1, 2, 3], "Maze number must be 1, 2, or 3."

    # load data
    #---------------------------------------------------------------------------------------------------------------
    root_dir = Path(__file__).parents[2]
    neg_log_likelihoods_total = {}
    for dataset in COMMON_FIELDS.keys():
        data_dir = f'{root_dir}/regression_results/is_it_markovian_results/{dataset}/maze_{maze_number}_regression_model.pt'
        data = torch.load(data_dir)
        neg_log_likelihoods = data['neg_log_likelihoods']
        neg_log_likelihoods_total[dataset] = neg_log_likelihoods

    # plot
    #---------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(3, 2, figsize=(10, 12), sharex=True, sharey=True)
    ax = np.array(ax)
    for i, dataset in enumerate(COMMON_FIELDS.keys()):
        neg_log_likelihoods = neg_log_likelihoods_total[dataset]
        unique_predictability = neg_log_likelihoods[..., 2:-1] - neg_log_likelihoods[..., -1:]
        unique_predictability_mean = unique_predictability.mean(dim=0)
        unique_predictability_std = unique_predictability.std(dim=0)

        x = np.arange(len(unique_predictability_mean))  # don't plot constant and
        y = unique_predictability_mean  # select the regressors to plot
        yerr = unique_predictability_std

        ax[i // 2, i % 2].bar(x, y, yerr=yerr, capsize=5)
        ax[i // 2, i % 2].set_title(dataset)
        ax[i // 2, i % 2].set_xticks(x)
        # ax[i // 2, i % 2].set_ylabel('Unique Predictability')
        # ax[i // 2, i % 2].set_ylim(bottom=0)
        ax[i // 2, i % 2].axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax[i // 2, i % 2].spines[['top', 'right']].set_visible(False)

        # statistical significance
        p_test = ttest_1samp(unique_predictability, 0)
        pvals = p_test.pvalue
        stats = p_test.statistic

        for j in range(len(unique_predictability_mean)):
            r_idx = j
            if stats[r_idx] > 0:
                if pvals[r_idx] <= 0.001:
                    ax[i // 2, i % 2].text(j, y[j] + yerr[j] + 0.001, '***', fontsize=12, color='black', ha='center', va='bottom')
                elif pvals[r_idx] <= 0.01:
                    ax[i // 2, i % 2].text(j, y[j] + yerr[j] + 0.001, '**', fontsize=12, color='black', ha='center', va='bottom')
                elif pvals[r_idx] <= 0.05:
                    ax[i // 2, i % 2].text(j, y[j] + yerr[j] + 0.001, '*', fontsize=12, color='black', ha='center', va='bottom')

    ax[2, 1].axis('off')
    ax[1, 1].tick_params(axis='x', labelbottom=True)
    ax[1, 0].set_ylabel('Unique Predictability')
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(f"{output_dir}/maze_{maze_number}.pdf", bbox_inches='tight')
        print(f"Figure saved to {output_dir}/maze_{maze_number}.pdf")
    plt.show()
    return


def plot_nonmarkovianness_all_mazes(output_dir=None):

    # load data
    #---------------------------------------------------------------------------------------------------------------
    root_dir = Path(__file__).parents[2]
    neg_log_likelihoods_total = defaultdict(list)
    for dataset in COMMON_FIELDS.keys():
        for maze_number in [1, 2, 3]:
            data_dir = f'{root_dir}/regression_results/is_it_markovian_results/{dataset}/maze_{maze_number}_regression_model.pt'
            data = torch.load(data_dir)
            neg_log_likelihoods = data['neg_log_likelihoods']
            neg_log_likelihoods_total[dataset].append(neg_log_likelihoods)
        neg_log_likelihoods_total[dataset] = torch.cat(neg_log_likelihoods_total[dataset], dim=0)

    # plot
    #---------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(3, 2, figsize=(10, 12), sharex=True, sharey=True)
    ax = np.array(ax)
    for i, dataset in enumerate(COMMON_FIELDS.keys()):
        neg_log_likelihoods = neg_log_likelihoods_total[dataset]
        unique_predictability = neg_log_likelihoods[..., 2:-1] - neg_log_likelihoods[..., -1:]
        unique_predictability_mean = unique_predictability.mean(dim=0)
        unique_predictability_std = unique_predictability.std(dim=0)

        x = np.arange(len(unique_predictability_mean))  # don't plot constant and
        y = unique_predictability_mean  # select the regressors to plot
        yerr = unique_predictability_std

        ax[i // 2, i % 2].bar(x, y, yerr=yerr, capsize=5)
        ax[i // 2, i % 2].set_title(dataset)
        ax[i // 2, i % 2].set_xticks(x)
        # ax[i // 2, i % 2].set_ylabel('Unique Predictability')
        # ax[i // 2, i % 2].set_ylim(bottom=0)
        ax[i // 2, i % 2].axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax[i // 2, i % 2].spines[['top', 'right']].set_visible(False)

        # statistical significance
        # need to think through this more carefully, for now, only plot for each maze
        # pvals = 1- (unique_predictability > 0.0).float().mean(dim=0).numpy()
        p_test = ttest_1samp(unique_predictability, 0)
        pvals = p_test.pvalue
        stats = p_test.statistic

        for j in range(len(unique_predictability_mean)):
            r_idx = j
            if stats[r_idx] > 0:
                if pvals[r_idx] <= 0.001:
                    ax[i // 2, i % 2].text(j, y[j] + yerr[j] + 0.001, '***', fontsize=12, color='black', ha='center', va='bottom')
                elif pvals[r_idx] <= 0.01:
                    ax[i // 2, i % 2].text(j, y[j] + yerr[j] + 0.001, '**', fontsize=12, color='black', ha='center', va='bottom')
                elif pvals[r_idx] <= 0.05:
                    ax[i // 2, i % 2].text(j, y[j] + yerr[j] + 0.001, '*', fontsize=12, color='black', ha='center', va='bottom')

    ax[2, 1].axis('off')
    ax[1, 1].tick_params(axis='x', labelbottom=True)
    ax[1, 0].set_ylabel('Unique Predictability')
    plt.tight_layout()
    # if output_dir is not None:
    #     plt.savefig(f"{output_dir}/all_mazes.pdf", bbox_inches='tight')
    #     print(f"Figure saved to {output_dir}/all_mazes.pdf")
    plt.show()
    return


if __name__ == "__main__":
    root_dir = Path(__file__).parents[2]
    output_dir = f"{root_dir}/figures/is_it_markovian_figures"
    os.makedirs(output_dir, exist_ok=True)

    plot_nonmarkovianness_all_mazes(output_dir=output_dir)
    # for maze_number in [1, 2, 3]:
    #     plot_nonmarkovianness(maze_number, output_dir)