from pathlib import Path
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast

from lowrank_lmdp.model import LowRankLMDP_HMM
from datahelper.load_data import get_data
from mazehelper.transition_matrix_functions import location_action_adjacency_matrix_from_maze_id
from mazehelper.plotting_functions import plot_policy_max


if __name__ == "__main__":
    root_dir = Path(__file__).parents[2]
    config_hash = "53a9cc28" #8c8d0f67 #9079a8dd #37373c66
    dataset = 'dijkstra_agents'
    maze_number = 1
    subject_ID = 2
    kfold = None

    kfold_suffix = f"_kfold_{kfold}" if kfold is not None else ""
    # load routes

    config_path = f'{root_dir}/inferred_routes/lowrank_lmdp_inferred/temp/{dataset}/{subject_ID}/Maze{maze_number}{kfold_suffix}/config_{config_hash}.json'
    model_path = f'{root_dir}/inferred_routes/lowrank_lmdp_inferred/temp/{dataset}/{subject_ID}/Maze{maze_number}{kfold_suffix}/best_{config_hash}.pt'
    # Load the config
    with open(config_path, "r") as f:
        params = json.load(f)

    model = LowRankLMDP_HMM(
        n_routes=params['n_routes'],
        n_locs=49,
        n_acts_per_loc=4,
        cognitive_constant=params['cognitive_constant'],
        action_cost=params['action_cost'],
        reward_value=params['reward_value'],
        route_entropy_param=params['route_entropy_param'],
        action_entropy_param=params['action_entropy_param'],
        noise=params['noise'],
        noise_decay=params['noise_decay'],
        adjacency_matrix=location_action_adjacency_matrix_from_maze_id(maze_id=params['maze_number']),
        lr=params['lr'],
    )
    tmp = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(tmp)



    data = get_data(dataset=params['dataset'], maze_number=params['maze_number'], query={"subject_ID": subject_ID, 'trial_phase': "navigation"})
    data = data[data.pos_idx != data.reward_idx]  # remove all reward positions
    x = torch.tensor(data['pos_idx'].to_numpy(), dtype=torch.long)
    a = torch.tensor(data['action_class'].to_numpy(), dtype=torch.long)
    r = torch.tensor(data['reward_idx'].to_numpy(), dtype=torch.long)

    model.eval()
    model.calculate_transition_matrix_and_policies(validation=True)
    log_prob_v, log_start_v, accuracy_v = model.E_step(x, a, r)
    log_likelihood = model.log_likelihood_under_trained_params(x, a, r)

    print(f"Log Likelihood of data under model: {log_likelihood:.2f}")
    print(f"accuracy: {accuracy_v*100:.2f}%")

    routes = F.normalize(model.R.sigmoid().detach().cpu(), p=1, dim=-1)
    routes_prior = F.normalize(model.pi.sigmoid().detach().cpu(), p=1, dim=-1)

    num_routes = routes.shape[0]
    ncols = 4  # Choose how many columns you want in your grid
    nrows = int(np.ceil(num_routes / ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))

    # Flatten axs for easy indexing, even if there's only one row or one plot
    axs = np.array(axs).reshape(-1)

    for i in range(num_routes):
        ax = axs[i]
        plt.sca(ax)  # Set the current axis for plot_policy_max
        plot_policy_max(
            routes[i].reshape(4, 49).T,
            maze_number=maze_number,
            cmap='Greens'
        )
        ax.set_title(f"Route {i + 1} (prior: {routes_prior[i]:.2f})")

    # Hide any unused subplots
    for j in range(num_routes, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()


    # plot real routes
    if dataset in ('lmdp_agents', 'dijkstra_agents'):
        root_dir = Path(__file__).parents[2]
        routes_df = pd.read_csv(f'{root_dir}/data/synthetic_data/{dataset[:-7]}/{dataset}_meta_data_{maze_number}.csv')
        routes_df['routes'] = routes_df['routes'].apply(ast.literal_eval)
        real_routes = routes_df.loc[routes_df.subject_ID == subject_ID, 'routes'].iloc[0]
        real_routes = torch.tensor(real_routes, dtype=torch.float32)

        # plot real routes
        num_real_routes = real_routes.shape[0]
        ncols_real = 4  # Choose how many columns you want in your grid
        nrows_real = int(np.ceil(num_real_routes / ncols_real))

        fig_real, axs_real = plt.subplots(nrows=nrows_real, ncols=ncols_real, figsize=(4 * ncols_real, 4 * nrows_real))
        axs_real = np.array(axs_real).reshape(-1)

        for i in range(num_real_routes):
            ax = axs_real[i]
            plt.sca(ax)  # Set the current axis for plot_policy_max
            plot_policy_max(
                real_routes[i].reshape(4, 49).T,
                maze_number=maze_number,
            )
            ax.set_title(f"Real Route {i + 1}")
        for j in range(num_real_routes, len(axs_real)):
            axs_real[j].axis('off')
        plt.tight_layout()
        plt.show()




    # for i in range(routes.shape[0]):
    #     plot_policy_max(
    #         routes[i].reshape(4, 49).t(),
    #         maze_number=maze_number,
    #     )
    #     plt.title(f"Route {i+1} (prior: {routes_prior[i]:.2f})")
    #     plt.show()




