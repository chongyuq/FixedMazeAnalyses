import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from scipy.stats import ttest_1samp, permutation_test
from pathlib import Path
from regressionhelper.constants import *
from datahelper.fields import COMMON_FIELDS
from regressionhelper.regression_loglikelihood_funcs import neg_log_likelihood


def plot_routeyness(maze_number, regressors, regressors_plot, output_dir=None):
    assert maze_number in [1, 2, 3], "Maze number must be 1, 2, or 3."
    assert all(r in REGRESSORS for r in regressors), f"Regressors must be in {list(REGRESSORS)}."
    assert all(r in REGRESSORS for r in regressors), f"Regressors plot keys must be in {REGRESSORS}."
    colour = np.array([REGRESSOR_COLOURS[r] for r in regressors])
    regressor_plot_names = [REGRESSOR_PLOT_NAMES[r] for r in regressors_plot]
    aliases = [REGRESSOR_ALIASES[r] for r in regressors]
    folder_name = '_'.join(aliases)

    # load data
    #---------------------------------------------------------------------------------------------------------------
    root_dir = Path(__file__).parents[2]
    neg_log_likelihoods_total = {}
    for dataset in COMMON_FIELDS.keys():
        data_dir = f'{root_dir}/regression_results/is_it_using_routes_results/{dataset}/{folder_name}/maze_{maze_number}_regression_model.pt'
        data = torch.load(data_dir)
        neg_log_likelihoods = data['neg_log_likelihoods']
        neg_log_likelihoods_total[dataset] = neg_log_likelihoods

    # plot
    #---------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(3, 2, figsize=(10, 12), sharex=True, sharey=True)
    ax = np.array(ax)
    for i, dataset in enumerate(COMMON_FIELDS.keys()):
        neg_log_likelihoods = neg_log_likelihoods_total[dataset]
        unique_predictability = neg_log_likelihoods[..., 1:-1] - neg_log_likelihoods[..., -1:]
        unique_predictability_mean = unique_predictability.mean(dim=(0, 1))
        unique_predictability_std = unique_predictability.mean(dim=0).std(dim=0)
        x = np.arange(len(regressors_plot))  # don't plot constant and
        y = unique_predictability_mean[[regressors.index(r) for r in regressors_plot]]  # select the regressors to plot
        yerr = unique_predictability_std[[regressors.index(r) for r in regressors_plot]]

        ax[i // 2, i % 2].bar(x, y, yerr=yerr, color=colour, capsize=5)
        ax[i // 2, i % 2].set_title(dataset)
        ax[i // 2, i % 2].set_xticks(x)
        ax[i // 2, i % 2].set_xticklabels(regressor_plot_names, rotation=45, ha='right')
        # ax[i // 2, i % 2].set_ylabel('Unique Predictability')
        # ax[i // 2, i % 2].set_ylim(bottom=0)
        ax[i // 2, i % 2].axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax[i // 2, i % 2].spines[['top', 'right']].set_visible(False)

        # statistical significance
        p_test = ttest_1samp(unique_predictability.mean(dim=0), 0)
        pvals = p_test.pvalue
        stats = p_test.statistic

        for j, r in enumerate(regressors_plot):
            r_idx = regressors_plot.index(r)
            if stats[r_idx] > 0:
                if pvals[r_idx] <= 0.001:
                    ax[i // 2, i % 2].text(j, y[j] + yerr[j] + 0.01, '***', fontsize=12, color='black', ha='center', va='bottom')
                elif pvals[r_idx] <= 0.01:
                    ax[i // 2, i % 2].text(j, y[j] + yerr[j] + 0.01, '**', fontsize=12, color='black', ha='center', va='bottom')
                elif pvals[r_idx] <= 0.05:
                    ax[i // 2, i % 2].text(j, y[j] + yerr[j] + 0.01, '*', fontsize=12, color='black', ha='center', va='bottom')

    ax[2, 1].axis('off')
    ax[1, 1].tick_params(axis='x', labelbottom=True)
    ax[1, 0].set_ylabel('Unique Predictability')
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(f"{output_dir}/maze_{maze_number}_routeyness_{folder_name}.pdf")
        print(f"Figure saved to {output_dir}/maze_{maze_number}_routeyness_{folder_name}.pdf")
    plt.show()
    return


if __name__ == "__main__":
    root_dir = Path(__file__).parents[2]
    output_dir = f"{root_dir}/figures/is_it_using_routes_figures"
    os.makedirs(output_dir, exist_ok=True)
    regressors = ['vector', 'optimal', 'pca_route', 'pca_route_planning', 'habit', 'forward', 'reverse']
    regressors_plot = ['vector', 'optimal', 'pca_route', 'pca_route_planning', 'habit', 'forward', 'reverse']

    for maze_number in [1, 2, 3]:
        plot_routeyness(maze_number, regressors,  regressors_plot, output_dir)