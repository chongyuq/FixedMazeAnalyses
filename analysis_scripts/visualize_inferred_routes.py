from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from mazehelper.plotting_functions import plot_policy_max


if __name__ == "__main__":
    root_dir = Path(__file__).parents[1]
    config_hash = "ed43d6d5"
    maze_number = 2
    subject_ID = 3
    # load routes

    tmp = torch.load(f'{root_dir}/inferred_routes/lowrank_lmdp_inferred/temp/lmdp_agents/{subject_ID}/Maze{maze_number}/best_{config_hash}.pt', map_location=torch.device('cpu'))
    routes = F.normalize(tmp['R'].sigmoid().detach().cpu(), dim=-1)
    routes_prior = F.normalize(tmp['pi'].sigmoid().detach().cpu(), dim=-1)

    for i in range(routes.shape[0]):
        plot_policy_max(
            routes[i].reshape(4, 49).t(),
            maze_number=maze_number,
            scale=10
        )
        plt.title(f"Route {i+1} (prior: {routes_prior[i]:.2f})")
        plt.show()