from synthetic_data_generate.agent_generation_funcs import generate_synthetic_agents


if __name__ == "__main__":
    for maze in [1, 2, 3]:
        generate_synthetic_agents(maze_number=maze, n_agents=6)

    print("All synthetic agents and data generated successfully.")