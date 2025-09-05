import subprocess
from pathlib import Path
from datahelper.reset_data import drop_all_collections


if __name__ == '__main__':
    fresh_run = False # this will rerun everything if set to True
    # note that if fresh_run is set to True, it will drop all collections in the database, the agents are
    # different every time, so if you want to keep the agents in the database, set fresh_run to False or save the current agents
    if fresh_run:
        drop_all_collections()
    rerun_pcs = False # this will rerun pcs if set to True
    rerun_regression = False # this will rerun regression if set to True

    root_dir = Path(__file__).parents[2]
    # ----------------- upload data to mongodb if not present
    subprocess.run(['python', f'{root_dir}/analysis_scripts/move_data_to_mongodb.py'])

    # ----------------- check for optimal policy
    if not Path(f'{root_dir}/data/synthetic_data/optimal/maze1.npy').exists() or \
       not Path(f'{root_dir}/data/synthetic_data/optimal/maze2.npy').exists() or\
       not Path(f'{root_dir}/data/synthetic_data/optimal/maze3.npy').exists() or fresh_run:
        subprocess.run(['python', f'{root_dir}/analysis_scripts/get_optimal_policy.py'])

    # ------------------ check for general statistics
    if not Path(f'{root_dir}/data/behaviour/summary_statistics/time_at_node.csv').exists() or \
       not Path(f'{root_dir}/data/behaviour/summary_statistics/trial_session_ITI_steps.csv').exists() or fresh_run:
        subprocess.run(['python', f'{root_dir}/analysis_scripts/statistics_on_time_on_node_trial_length.py'])

    if not Path(f'{root_dir}/data/behaviour/summary_statistics/vector_optimal_forward_proportions.csv').exists() or \
       not Path(f'{root_dir}/data/behaviour/summary_statistics/vector_optimal_proportions.csv').exists() or fresh_run:
        subprocess.run(['python', f'{root_dir}/analysis_scripts/vector_optimal_straight_proportions.py'])

    # ------------------ check for all agents in directory
    if not Path(f'{root_dir}/data/synthetic_data/markovian/markovian_agents_maze_1.csv').exists() or \
       not Path(f'{root_dir}/data/synthetic_data/markovian/markovian_agents_maze_2.csv').exists() or \
       not Path(f'{root_dir}/data/synthetic_data/markovian/markovian_agents_maze_3.csv').exists() or \
       not Path(f'{root_dir}/data/synthetic_data/non_markovian/non_markovian_agents_maze_1.csv').exists() or \
       not Path(f'{root_dir}/data/synthetic_data/non_markovian/non_markovian_agents_maze_2.csv').exists() or \
       not Path(f'{root_dir}/data/synthetic_data/non_markovian/non_markovian_agents_maze_3.csv').exists() or \
       not Path(f'{root_dir}/data/synthetic_data/lmdp/lmdp_agents_maze_1.csv').exists() or \
       not Path(f'{root_dir}/data/synthetic_data/lmdp/lmdp_agents_maze_2.csv').exists() or \
       not Path(f'{root_dir}/data/synthetic_data/lmdp/lmdp_agents_maze_3.csv').exists() or \
       not Path(f'{root_dir}/data/synthetic_data/dijkstra/dijkstra_agents_maze_1.csv').exists() or \
       not Path(f'{root_dir}/data/synthetic_data/dijkstra/dijkstra_agents_maze_2.csv').exists() or \
       not Path(f'{root_dir}/data/synthetic_data/dijkstra/dijkstra_agents_maze_3.csv').exists() or fresh_run:
        subprocess.run(['python', f'{root_dir}/synthetic_agents_generating_scripts/generate_all_agents.py'])

    # ------------------ check for agents in database
    subprocess.run(['python', f'{root_dir}/analysis_scripts/synthetic_agents_generating_scripts/move_data_to_mongodb.py'])

    # ----------------- check for habits
    if not Path(f'{root_dir}/habits/habits_dijkstra_agents.pt').exists() or \
       not Path(f'{root_dir}/habits/habits_lmdp_agents.pt').exists() or \
       not Path(f'{root_dir}/habits/habits_markovian_agents.pt').exists() or \
       not Path(f'{root_dir}/habits/habits_non_markovian_agents.pt').exists() or \
       not Path(f'{root_dir}/habits/habits_mice_behaviour.pt').exists() or fresh_run:
        subprocess.run(['python', f'{root_dir}/analysis_scripts/generate_all_habits.py'])

    # ------------------ check for PCs
    if rerun_pcs or fresh_run:
        subprocess.run(['python', f'{root_dir}/analysis_scripts/route_inference_scripts/generate_all_pcs.py'])

    # ------------------ run regression analyses
    if rerun_regression or fresh_run:
        subprocess.run(['python', f'{root_dir}/analysis_scripts/is_it_using_routes_scripts/regression_script.py'])

    # ------------------ final plots
    subprocess.run(['python', f'{root_dir}/analysis_scripts/is_it_using_routes_scripts/generate_plots.py'])
