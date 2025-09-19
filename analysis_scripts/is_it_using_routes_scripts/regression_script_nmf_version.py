from pathlib import Path
import torch
from fontTools.misc.cython import returns

from regressionhelper.regression_pipeline import run_regression_pipeline
from datahelper.fields import COMMON_FIELDS
import pandas as pd


if __name__ == "__main__":
    # regression analysis on whether the agent is using vector, optimal, or route strategies
    # ----------------------------------------------------------------------------------------------
    datasets = list(COMMON_FIELDS.keys())
    # datasets = ['non_markovian_agents']

    for maze_number in range(1, 4):
        for dataset in datasets:
            run_regression_pipeline(
                root_dir = Path(__file__).parents[2],
                maze_number = maze_number,
                dataset = dataset,
                regressors = ["vector", "optimal", "nmf_route", "nmf_route_planning", "habit", "forward", "reverse"],
                bootstraps = None,  # this will the regression per subject
                save_model = True
            )
