import subprocess
from pathlib import Path
from nonmarkovianhelper.generate_plots import plot_nonmarkovianness
import os


if __name__ == '__main__':
    rerun_regression = False # this will rerun regression if set to True

    root_dir = Path(__file__).parents[2]

    # ------------------ run regression analyses
    if rerun_regression:
        subprocess.run(['python', f'{root_dir}/analysis_scripts/is_it_markovian_scripts/regression_script.py'])

    # ------------------ final plots
    output_dir = f"{root_dir}/figures/is_it_markovian_figures"
    os.makedirs(output_dir, exist_ok=True)

    for maze_number in [1, 2, 3]:
        plot_nonmarkovianness(maze_number, output_dir)

