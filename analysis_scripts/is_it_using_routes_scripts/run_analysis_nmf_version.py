import subprocess
from pathlib import Path
from datahelper.reset_data import drop_all_collections
from regressionhelper.generate_plots import plot_routeyness
import os


if __name__ == '__main__':
    rerun_nmfs = False # this will rerun pcs if set to True
    rerun_regression = False # this will rerun regression if set to True

    root_dir = Path(__file__).parents[2]
    # ------------------ check for PCs
    if rerun_nmfs :
        subprocess.run(['python', f'{root_dir}/analysis_scripts/route_inference_scripts/generate_all_nmfs.py'])

    # ------------------ run regression analyses
    if rerun_regression:
        subprocess.run(['python', f'{root_dir}/analysis_scripts/is_it_using_routes_scripts/regression_script_nmf_version.py'])

    # ------------------ final plots
    regressors = ['vector', 'optimal', 'nmf_route', 'nmf_route_planning', 'habit', 'forward', 'reverse']
    # regressors_plot = ['vector', 'optimal', 'nmf_route', 'nmf_route_planning', 'habit', 'forward', 'reverse']
    regressors_plot = ['vector', 'optimal','nmf_route_planning', 'forward']

    output_dir = f"{root_dir}/figures/is_it_using_routes_figures"
    os.makedirs(output_dir, exist_ok=True)
    for maze_number in [1, 2, 3]:
        plot_routeyness(maze_number, regressors,  regressors_plot, output_dir)

