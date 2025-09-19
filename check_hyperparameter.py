import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from datahelper.load_data import load_subject_IDs

if __name__ == '__main__':
    root_dir = Path(__file__).parents[0]
    maze_number = 2
    dataset = 'lmdp_agents'
    subject_IDs = load_subject_IDs(dataset)
    for subject_ID in subject_IDs:
        leadership_df = pd.read_csv(f'{root_dir}/inferred_routes/lowrank_lmdp_inferred/temp/{dataset}_hpo/{subject_ID}/Maze{maze_number}/hpo_results/leaderboard.csv')
        true_hpos = pd.read_csv(f'{root_dir}/data/synthetic_data/{dataset[:-7]}/{dataset}_meta_data_{maze_number}.csv')
        true_hpos = true_hpos.set_index('subject_ID').loc[subject_ID]
        print(f"True Hyperparameters for Subject {subject_ID} Maze {maze_number}:")
        if dataset == 'dijkstra_agents':
            print(true_hpos[['cognitive_constant', 'n_routes']])
        if dataset == 'lmdp_agents':
            print(true_hpos[['cognitive_constant', 'action_cost', 'n_routes']])

        leadership_df['cognitive_constant_round'] = leadership_df['params_cognitive_constant'].round()
        leadership_df_cognitive_mean = leadership_df.groupby('cognitive_constant_round').value.mean().reset_index()

        leadership_df['action_cost_round'] = leadership_df['params_action_cost'].round(2)
        leadership_df_action_cost_mean = leadership_df.groupby('action_cost_round').value.mean().reset_index()

        leadership_df['entropy_round'] = leadership_df['params_route_entropy_param'].round(2)
        leadership_df_entropy_mean = leadership_df.groupby('entropy_round').value.mean().reset_index()

        leadership_all_mean = leadership_df.groupby(['action_cost_round', 'cognitive_constant_round']).value.mean().reset_index()

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Subject {subject_ID} Maze {maze_number} - Parameter Sweeps', fontsize=16)

        # Plot 1: Route Entropy Parameter vs Value
        axs[0, 0].scatter(leadership_df['params_route_entropy_param'], leadership_df['value'], alpha=0.1)
        axs[0, 0].plot(leadership_df_entropy_mean['entropy_round'], leadership_df_entropy_mean['value'], color='red')
        axs[0, 0].set_xlabel('Route Entropy Parameter')
        axs[0, 0].set_ylabel('Value')
        axs[0, 0].set_title(f'Route Entropy vs Value - true value: {true_hpos.n_routes}')

        # Plot 2: Cognitive Constant vs Value
        axs[0, 1].scatter(leadership_df['params_cognitive_constant'], leadership_df['value'], alpha=0.1)
        axs[0, 1].plot(leadership_df_cognitive_mean['cognitive_constant_round'], leadership_df_cognitive_mean['value'], color='red')
        axs[0, 1].set_xlabel('Cognitive Constant')
        axs[0, 1].set_ylabel('Value')
        axs[0, 1].set_title(f'Cognitive Constant vs Value - true value: {true_hpos.cognitive_constant}')

        # Plot 3: Action Cost vs Value
        axs[1, 0].scatter(leadership_df['params_action_cost'], leadership_df['value'], alpha=0.1)
        axs[1, 0].plot(leadership_df_action_cost_mean['action_cost_round'], leadership_df_action_cost_mean['value'], color='red')
        axs[1, 0].set_xlabel('Action Cost')
        axs[1, 0].set_ylabel('Value')
        if dataset == 'lmdp_agents':
            axs[1, 0].set_title(f'Action Cost vs Value - true value: {true_hpos.action_cost}')
        else:
            axs[1, 0].set_title('Action Cost vs Value')

        # Plot 4: Action Cost vs Value colored by Cognitive Constant
        sc = axs[1, 1].scatter(
            leadership_df['params_action_cost'],
            leadership_df['params_cognitive_constant'],
            c=leadership_df['value'],
            cmap='plasma',
            alpha=0.5,
            s=500
        )
        axs[1, 1].set_xlabel('Action Cost')
        axs[1, 1].set_ylabel('Value')
        axs[1, 1].set_title('Action Cost vs Cognitive Constant (coloured by Value)')
        cbar = fig.colorbar(sc, ax=axs[1, 1], label='Cognitive Constant')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

