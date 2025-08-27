from pathlib import Path
import torch
from fontTools.misc.cython import returns

from regressionhelper.regression_pipeline import run_regression_pipeline
import pandas as pd


if __name__ == "__main__":
    # regression analysis on whether the agent is using vector, optimal, or straight strategies
    # ----------------------------------------------------------------------------------------------
    for maze_number in range(1, 4):
        run_regression_pipeline(
            root_dir = Path(__file__).parents[1],
            maze_number = maze_number,
            dataset = "mice_behaviour",
            regressors = ["vector", "optimal", "forward"],
            bootstraps = None,
            save_model = True
        )

        run_regression_pipeline(
            root_dir = Path(__file__).parents[1],
            maze_number = maze_number,
            dataset = "mice_behaviour",
            regressors = ["vector", "optimal"],
            bootstraps = None,
            save_model = True
        )

    # load the saved models and find regression coefficients for the vector, optimal, forward regression
    # ----------------------------------------------------------------------------------------------
    root_dir = Path(__file__).parents[1]
    n_subjects = 6  # number of subjects in the mice_behaviour dataset

    n_regressors = 4  # vector, optimal, forward
    vec_opt_fwd_coefs = []
    for maze_number in range(1, 4):
        data = torch.load(f"{root_dir}/regression_results/is_it_using_routes_results/mice_behaviour/vec_opt_fwd/maze_{maze_number}_regression_model.pt")
        coefs = data['coefs'][..., -1, :]  # (n_folds, n_bootstraps, n_regressors) # taking the last model
        vec_opt_fwd_coefs.append(coefs)
    rows = []
    regressors = ["constant", "vector", "optimal", "forward"]
    for maze_idx, tensor in enumerate(vec_opt_fwd_coefs, start=1):
        n_days = tensor.shape[0]
        for day in range(n_days):
            for subject in range(n_subjects):
                row = {
                    "maze_number": maze_idx,
                    "day_on_maze": day + 1,
                    "subject_ID": subject
                }
                for r in range(n_regressors):
                    row[f"{regressors[r]}"] = tensor[day, subject, r].item()
                rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    df.to_csv(f"{root_dir}/data/behaviour/summary_statistics/vector_optimal_forward_proportions.csv", index=False)

    # Preview
    # ----------------------------------------------------------------------------------------------
    print(df.head())


    # load the saved models and find regression coefficients for the vector, optimal regression
    # ----------------------------------------------------------------------------------------------
    n_regressors = 3  # vector, optimal
    vec_opt_coefs = []
    for maze_number in range(1, 4):
        data = torch.load(f"{root_dir}/regression_results/is_it_using_routes_results/mice_behaviour/vec_opt/maze_{maze_number}_regression_model.pt")
        coefs = data['coefs'][..., -1, :]  # (n_folds, n_bootstraps, n_regressors) # taking the last model
        vec_opt_coefs.append(coefs)
    rows = []
    regressors = ["constant", "vector", "optimal"]
    for maze_idx, tensor in enumerate(vec_opt_coefs, start=1):
        n_days = tensor.shape[0]
        for day in range(n_days):
            for subject in range(n_subjects):
                row = {
                    "maze_number": maze_idx,
                    "day_on_maze": day + 1,
                    "subject_ID": subject
                }
                for r in range(n_regressors):
                    row[f"{regressors[r]}"] = tensor[day, subject, r].item()
                rows.append(row)
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    df.to_csv(f"{root_dir}/data/behaviour/summary_statistics/vector_optimal_proportions.csv", index=False)
    # Preview
    # ----------------------------------------------------------------------------------------------
    print(df.head())

