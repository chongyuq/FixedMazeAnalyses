import json, subprocess, sys, os, tempfile, shlex, math
import optuna
from types import SimpleNamespace
from pathlib import Path
import torch
from lmdphelper.train_entry import run_once
from lmdphelper.callback import leaderboard_callback
from lmdphelper.hp_objective import objective
from datahelper.load_data import load_subject_IDs
import argparse


def int_or_none(value):
    if value in (None, "", "None"):
        return None
    try:
        # try parsing as int first
        return int(value)
    except ValueError:
        # if it's a float like "10.0", cast safely
        f = float(value)
        if f.is_integer():
            return int(f)
        raise ValueError(f"Invalid integer or None value: {value}")


if __name__ == "__main__":
    root_dir = Path(__file__).parents[2]
    # main parser
    ap = argparse.ArgumentParser()
    # data / split
    ap.add_argument('--dataset', type=str, default='lmdp_agents')
    ap.add_argument('--maze_number', type=int, default=1)
    ap.add_argument('--subject_index', type=int, default=5)
    ap.add_argument('--kfold', type=int_or_none, default=None)
    ap.add_argument('--sample_ratio', type=float, default=1)
    # model params
    ap.add_argument('--n_routes', type=int, default=4)
    ap.add_argument('--cognitive_constant', type=float, default=28.28)  # 11.51  28.283460365070333,0.15027235205849984
    ap.add_argument('--action_cost', type=float, default=0.1502)
    ap.add_argument('--reward_value', type=float, default=1.07)  # 1.07
    ap.add_argument('--route_entropy_param', type=float, default=0)
    ap.add_argument('--action_entropy_param', type=float, default=0)
    ap.add_argument('--noise', type=float, default=0)
    ap.add_argument('--noise_decay', type=float, default=0.99)
    # training
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--eval_every', type=int, default=50)
    ap.add_argument('--smooth_k', type=int, default=5)
    ap.add_argument('--log_every', type=int, default=10)
    # io
    ap.add_argument('--saving', type=int, default=1)
    ap.add_argument('--tensorboard', type=int, default=1)
    ap.add_argument('--output_dir', type=str, default=f'{root_dir}/inferred_routes/lowrank_lmdp_inferred/temp')

    # set defaults from config file
    default_config_file = f"{root_dir}/inferred_routes/lowrank_lmdp_inferred/configs/lowrank_lmdp_model_default.json"

    if os.path.exists(default_config_file):
        with open(default_config_file, 'r') as f:
            config = json.load(f)
        ap.set_defaults(**config)
    else:
        print(f"Config file {default_config_file} not found. Please download config file")

    # set optimal config path
    args, _ = ap.parse_known_args()
    optimal_config_file = (f"{root_dir}/inferred_routes/lowrank_lmdp_inferred/temp/"
                      f"{args.dataset}_hpo/{load_subject_IDs(args.dataset)[args.subject_index]}/"
                      f"Maze{args.maze_number}/hpo_results/best_config.json")

    if os.path.exists(optimal_config_file):
        with open(optimal_config_file, 'r') as f:
            optimal_config = json.load(f)
        ap.set_defaults(**optimal_config['params'])
        ap.set_defaults(route_entropy_param=0) # override to 0
    else:
        print(f"Config file {optimal_config_file} not found. Using default command line arguments.")

    args = ap.parse_args()

    print('Running with config: dataset {}, maze_number {}, subject_index {}, kfold {}, cognitive_constant {}, action_cost {}, route_entropy_param {}'.format(args.dataset, args.maze_number, args.subject_index, args.kfold, args.cognitive_constant, args.action_cost, args.route_entropy_param))
    test = run_once(args)