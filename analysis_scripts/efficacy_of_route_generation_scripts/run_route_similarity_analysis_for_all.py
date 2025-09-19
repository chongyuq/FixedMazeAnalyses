import subprocess


from pathlib import Path
from datahelper.fields import COMMON_FIELDS


if __name__ == '__main__':
    root_dir = Path(__file__).parents[2]
    datasets = ['lmdp_agents', 'dijkstra_agents']
    maze_numbers = [1, 2, 3]

    # ------------------ run regression analyses
    for dataset in datasets:
        for maze_number in maze_numbers:
            subprocess.run(['python', f'{root_dir}/analysis_scripts/efficacy_of_route_generation_scripts/route_similarities_analysis.py',
                            '--dataset', dataset,
                            '--mazes', str(maze_number)])
