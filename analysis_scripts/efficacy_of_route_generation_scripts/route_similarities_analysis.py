import pandas as pd
from pathlib import Path
import ast
import torch
import torch.nn.functional as F
from glob import glob
import numpy as np
import argparse
import os
from scipy.stats import ttest_rel, ttest_1samp

from datahelper.load_data import load_subject_IDs
from pcahelper.route_conversion_funcs import pc_to_route


def fisher_z(r):
    return 0.5 * (torch.log1p(r) - torch.log1p(-r))


# plot in significance stars
def significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


if __name__=='__main__':
    root_dir = Path(__file__).parents[2]
    output_dir = f'{root_dir}/figures/efficacy_of_route_generation_scripts'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dijkstra_agents', help='Dataset to analyze')
    parser.add_argument('--mazes', type=int, nargs='+', default=[2], help='Maze numbers to analyze')
    args, _ = parser.parse_known_args()
    dataset = args.dataset
    maze_numbers = args.mazes

    n_mazes = len(maze_numbers)
    print(f"Running for dataset: {dataset}, mazes: {maze_numbers}")

    # get real routes
    # ------------------------------------------------------------------------------
    real_routes_total = []
    no_of_routes_per_agent = []
    for maze_number in maze_numbers:
        for subject_ID in load_subject_IDs(dataset):
            routes_df = pd.read_csv(
                f'{root_dir}/data/synthetic_data/{dataset[:-7]}/{dataset}_meta_data_{maze_number}.csv')
            routes_df['routes'] = routes_df['routes'].apply(ast.literal_eval)
            real_routes = routes_df.loc[routes_df.subject_ID == subject_ID, 'routes'].iloc[0]
            real_routes = torch.tensor(real_routes, dtype=torch.float32)
            real_routes_total.append(real_routes)
            no_of_routes_per_agent.append(real_routes.shape[0])
    real_routes_total = torch.cat(real_routes_total)

    # get lmdp inferred routes
    # ------------------------------------------------------------------------------
    lmdp_inferred_routes_total = []
    lmdp_no_inferred_routes_per_agent = []
    inferred_routes_dir = f'{root_dir}/inferred_routes/lowrank_lmdp_inferred/temp/{dataset}'
    for maze_number in maze_numbers:
        for subject_ID in load_subject_IDs(dataset):
            file_name = glob(f'{inferred_routes_dir}/{subject_ID}/Maze{maze_number}/best_*.pt')[0]
            prior_and_routes = torch.load(file_name, map_location='cpu')
            inferred_route = F.normalize(prior_and_routes['R'].sigmoid(), p=1, dim=-1)
            lmdp_inferred_routes_total.append(inferred_route)
            lmdp_no_inferred_routes_per_agent.append(inferred_route.shape[0])
    lmdp_inferred_routes_total = torch.cat(lmdp_inferred_routes_total)

    # get pca inferred routes
    # ------------------------------------------------------------------------------
    pca_inferred_routes_total = []
    pca_no_inferred_routes_per_agent = []
    inferred_routes_dir = f'{root_dir}/inferred_routes/pca_inferred/history_True_future_True_combine_sum_normalize_True_alpha_0.1/{dataset}'
    for maze_number in maze_numbers:
        for subject_ID in load_subject_IDs(dataset):
            file_name = f'{inferred_routes_dir}/{subject_ID}/Maze{maze_number}_PCs.pt'
            variance_and_pcs = torch.load(file_name, map_location='cpu')
            # inferred_route = torch.cat([variance_and_pcs['pcs'][:, :3], -variance_and_pcs['pcs'][:, :3]], dim=-1).t()
            inferred_route = pc_to_route(variance_and_pcs['pcs'][:, :3]).t()
            pca_inferred_routes_total.append(inferred_route)
            pca_no_inferred_routes_per_agent.append(inferred_route.shape[0])
    pca_inferred_routes_total = torch.cat(pca_inferred_routes_total)

    # get nmf inferred routes
    # ------------------------------------------------------------------------------
    nmf_inferred_routes_total = []
    nmf_no_inferred_routes_per_agent = []
    inferred_routes_dir = f'{root_dir}/inferred_routes/nmf_inferred/history_True_future_True_combine_sum_normalize_True_alpha_0.1/{dataset}'
    for maze_number in maze_numbers:
        for subject_ID in load_subject_IDs(dataset):
            file_name = f'{inferred_routes_dir}/{subject_ID}/Maze{maze_number}_NMFs.pt'
            weight_and_nmf = torch.load(file_name, map_location='cpu')
            inferred_route = F.normalize(weight_and_nmf['nmfs'].t(), p=1, dim=-1)
            nmf_inferred_routes_total.append(inferred_route)
            nmf_no_inferred_routes_per_agent.append(inferred_route.shape[0])
    nmf_inferred_routes_total = torch.cat(nmf_inferred_routes_total)

    # get correlation matrix between real and inferred routes
    # ------------------------------------------------------------------------------
    real_routes_corr_with_real = torch.corrcoef(real_routes_total)
    real_routes_corr_with_lmdp_inferred = torch.corrcoef(torch.cat([real_routes_total, lmdp_inferred_routes_total]))[:real_routes_total.shape[0], real_routes_total.shape[0]:]
    real_routes_corr_with_pca_inferred = torch.corrcoef(torch.cat([real_routes_total, pca_inferred_routes_total]))[:real_routes_total.shape[0], real_routes_total.shape[0]:]
    real_routes_corr_with_nmf_inferred = torch.corrcoef(torch.cat([real_routes_total, nmf_inferred_routes_total]))[:real_routes_total.shape[0], real_routes_total.shape[0]:]

    # split correlation matrices into the appropriate chunks
    # ------------------------------------------------------------------------------
    real_routes_corr_with_real_row_split = torch.split(real_routes_corr_with_real, no_of_routes_per_agent, dim=0)
    real_routes_corr_with_real_chunks = [torch.split(x, no_of_routes_per_agent, dim=1) for x in real_routes_corr_with_real_row_split]
    real_routes_corr_with_other_real = torch.tensor([real_routes_corr_with_real_chunks[i][j].max(dim=-1)[0].mean() for i in range(len(no_of_routes_per_agent)) for j in range(i+1, i // 6 * 6 + 6)])

    real_routes_corr_with_lmdp_inferred_row_split = torch.split(real_routes_corr_with_lmdp_inferred, no_of_routes_per_agent, dim=0)
    real_routes_corr_with_lmdp_inferred_chunks = [torch.split(x, lmdp_no_inferred_routes_per_agent, dim=1) for x in real_routes_corr_with_lmdp_inferred_row_split]
    real_routes_corr_with_lmdp_inferred_mean = torch.tensor([real_routes_corr_with_lmdp_inferred_chunks[i][i].max(dim=-1)[0].mean() for i in range(len(no_of_routes_per_agent))])
    real_routes_corr_with_lmdp_inferred_other = torch.tensor([real_routes_corr_with_lmdp_inferred_chunks[i][j].max(dim=-1)[0].mean() for i in range(len(no_of_routes_per_agent)) for j in range(i // 6 * 6, i // 6 * 6 + 6) if i != j])

    real_routes_corr_with_pca_inferred_row_split = torch.split(real_routes_corr_with_pca_inferred, no_of_routes_per_agent, dim=0)
    real_routes_corr_with_pca_inferred_chunks = [torch.split(x, pca_no_inferred_routes_per_agent, dim=1) for x in real_routes_corr_with_pca_inferred_row_split]
    real_routes_corr_with_pca_inferred_mean = torch.tensor([real_routes_corr_with_pca_inferred_chunks[i][i].max(dim=-1)[0].mean() for i in range(len(no_of_routes_per_agent))])
    real_routes_corr_with_pca_inferred_other = torch.tensor([real_routes_corr_with_pca_inferred_chunks[i][j].max(dim=-1)[0].mean() for i in range(len(no_of_routes_per_agent)) for j in range(i // 6 * 6, i // 6 * 6 + 6) if i != j])

    real_routes_corr_with_nmf_inferred_row_split = torch.split(real_routes_corr_with_nmf_inferred, no_of_routes_per_agent, dim=0)
    real_routes_corr_with_nmf_inferred_chunks = [torch.split(x, nmf_no_inferred_routes_per_agent, dim=1) for x in real_routes_corr_with_nmf_inferred_row_split]
    real_routes_corr_with_nmf_inferred_mean = torch.tensor([real_routes_corr_with_nmf_inferred_chunks[i][i].max(dim=-1)[0].mean() for i in range(len(no_of_routes_per_agent))])
    real_routes_corr_with_nmf_inferred_other = torch.tensor([real_routes_corr_with_nmf_inferred_chunks[i][j].max(dim=-1)[0].mean() for i in range(len(no_of_routes_per_agent)) for j in range(i // 6 * 6, i // 6 * 6 + 6) if i != j])

    # print results
    # ------------------------------------------------------------------------------
    print(f"Correlation between real routes: {real_routes_corr_with_other_real.mean():.3f} ± {real_routes_corr_with_other_real.std():.3f}")
    print(f"Correlation between real and LMDP inferred routes (same agent): {real_routes_corr_with_lmdp_inferred_mean.mean():.3f} ± {real_routes_corr_with_lmdp_inferred_mean.std():.3f}")
    print(f"Correlation between real and LMDP inferred routes (different agents): {real_routes_corr_with_lmdp_inferred_other.mean():.3f} ± {real_routes_corr_with_lmdp_inferred_other.std():.3f}")
    print(f"Correlation between real and PCA inferred routes (same agent): {real_routes_corr_with_pca_inferred_mean.mean():.3f} ± {real_routes_corr_with_pca_inferred_mean.std():.3f}")
    print(f"Correlation between real and PCA inferred routes (different agents): {real_routes_corr_with_pca_inferred_other.mean():.3f} ± {real_routes_corr_with_pca_inferred_other.std():.3f}")
    print(f"Correlation between real and NMF inferred routes (same agent): {real_routes_corr_with_nmf_inferred_mean.mean():.3f} ± {real_routes_corr_with_nmf_inferred_mean.std():.3f}")
    print(f"Correlation between real and NMF inferred routes (different agents): {real_routes_corr_with_nmf_inferred_other.mean():.3f} ± {real_routes_corr_with_nmf_inferred_other.std():.3f}")

    # find if the algorithms can distinguish between agent and other agents
    # ------------------------------------------------------------------------------
    lmdp_diff = []
    for i in range(6):
        zt = fisher_z(real_routes_corr_with_lmdp_inferred_mean[i])
        for j in range(5):
            zo = fisher_z(real_routes_corr_with_lmdp_inferred_other[i * 5 + j])
            lmdp_diff.append(zt - zo)
    lmdp_diff = torch.tensor(lmdp_diff)
    _, p_val_lmdp = ttest_1samp(lmdp_diff, 0.0, alternative='greater')

    pca_diff = []
    for i in range(6):
        zt = fisher_z(real_routes_corr_with_pca_inferred_mean[i])
        for j in range(5):
            zo = fisher_z(real_routes_corr_with_pca_inferred_other[i * 5 + j])
            pca_diff.append(zt - zo)
    pca_diff = torch.tensor(pca_diff)
    _, p_val_pca = ttest_1samp(pca_diff, 0.0, alternative='greater')


    nmf_diff = []
    for i in range(6):
        zt = fisher_z(real_routes_corr_with_nmf_inferred_mean[i])
        for j in range(5):
            zo = fisher_z(real_routes_corr_with_nmf_inferred_other[i * 5 + j])
            nmf_diff.append(zt - zo)
    nmf_diff = torch.tensor(nmf_diff)
    _, p_val_nmf = ttest_1samp(nmf_diff, 0.0, alternative='greater')

    # plot results
    # ------------------------------------------------------------------------------
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.boxplot([
        real_routes_corr_with_other_real.numpy(),
        real_routes_corr_with_lmdp_inferred_mean.numpy(),
        real_routes_corr_with_lmdp_inferred_other.numpy(),
        real_routes_corr_with_pca_inferred_mean.numpy(),
        real_routes_corr_with_pca_inferred_other.numpy(),
        real_routes_corr_with_nmf_inferred_mean.numpy(),
        real_routes_corr_with_nmf_inferred_other.numpy()
    ], tick_labels=[
        'Real vs Real\n(different agents)',
        'Real vs LMDP\n(same agent)',
        'Real vs LMDP\n(different agents)',
        'Real vs PCA\n(same agent)',
        'Real vs PCA\n(different agents)',
        'Real vs NMF\n(same agent)',
        'Real vs NMF\n(different agents)'
    ])

    comparisons = [
        (2, 3),  # LMDP same vs different
        (4, 5),  # PCA same vs different
        (6, 7)   # NMF same vs different
    ]
    for (i, j), p_val in zip(comparisons, [p_val_lmdp, p_val_pca, p_val_nmf]):
        y_max = max(plt.gca().get_ylim()[1],
                    max(real_routes_corr_with_lmdp_inferred_mean.max().item(),
                        real_routes_corr_with_lmdp_inferred_other.max().item(),
                        real_routes_corr_with_pca_inferred_mean.max().item(),
                        real_routes_corr_with_pca_inferred_other.max().item(),
                        real_routes_corr_with_nmf_inferred_mean.max().item(),
                        real_routes_corr_with_nmf_inferred_other.max().item())) + 0.05
        plt.plot([i, i, j, j], [y_max - 0.02, y_max, y_max, y_max - 0.02], color='black')
        plt.text((i + j) * 0.5, y_max, significance_stars(p_val), ha='center', va='bottom', color='black')

    plt.ylabel('Mean Max Correlation')
    plt.title(f'{dataset.split("_")[0]} agents maze {", ".join([str(i) for i in maze_numbers])} route similarity')
    # plt.grid(axis='y')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs(f'{output_dir}/{dataset}', exist_ok=True)
    plt.savefig(f'{output_dir}/{dataset}/maze_{"_".join([str(i) for i in maze_numbers])}_similarity.pdf')
    plt.show()


