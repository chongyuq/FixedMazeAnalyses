import argparse
import os

from synthetic_data_generate.agent_generation_funcs import generate_synthetic_agents


if __name__ == "__main__":
    # Define arguments and directories
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int, default=6)

    args = parser.parse_args()
    n_agents = args.n_agents

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agent_types = ['non_markovian_agents', 'markovian_agents', 'lmdp_agents', 'dijkstra_agents']

    for maze in [1, 2, 3]:
        for agent_type in agent_types:
            for agent in range(n_agents):
                # Generate synthetic agents
                generate_synthetic_agents(root_dir, agent_type, maze, agent)

    print("All synthetic agents and data generated successfully.")