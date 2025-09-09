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


if __name__ == "__main__":
    root_dir = Path(__file__).parents[2]
    ap = argparse.ArgumentParser()

    # pruning policy
    ap.add_argument("--min_epochs", type=int, default=300)
    ap.add_argument("--max_epochs", type=int, default=2000)
    ap.add_argument("--n_startup_trials", type=int, default=6)
    ap.add_argument("--interval_steps", type=int, default=100)
    ap.add_argument("--timeout", type=int, default=3600*9.5)  # in seconds

    # data/ training context
    ap.add_argument("--dataset", type=str, default="lmdp_agents")
    ap.add_argument("--maze_number", type=int, default=2)
    ap.add_argument("--subject_index", type=int, default=2)
    ap.add_argument("--kfold", type=int, default=None)
    ap.add_argument("--sample_ratio", type=float, default=0.8)
    ap.add_argument("--n_routes", type=int, default=6)
    ap.add_argument("--reward_value", type=float, default=2.5)
    ap.add_argument("--action_entropy_param", type=float, default=0.2)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--noise_decay", type=float, default=1)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--smooth_k", type=int, default=5)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--saving", type=int, default=0)
    ap.add_argument("--tensorboard", type=int, default=0)

    args = ap.parse_args()
    subject_id = load_subject_IDs(args.dataset)[args.subject_index]
    kfold_suffix = f"_kfold_{args.kfold}" if args.kfold is not None else ""

    fixed = SimpleNamespace(
        dataset=args.dataset,
        maze_number=args.maze_number,
        subject_index=args.subject_index,
        kfold=args.kfold,
        sample_ratio=args.sample_ratio,
        n_routes=args.n_routes,
        reward_value=args.reward_value,
        action_entropy_param=args.action_entropy_param,
        noise=args.noise,
        noise_decay=args.noise_decay,
        lr=args.lr,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        eval_every=args.eval_every,
        smooth_k=args.smooth_k,
        log_every=args.log_every,
        saving=bool(args.saving),
        tensorboard=bool(args.tensorboard),
        local_root=f"{root_dir}/inferred_routes/lowrank_lmdp_inferred/temp/{args.dataset}_hpo/{subject_id}/Maze{args.maze_number}{kfold_suffix}"
    )


    sampler = optuna.samplers.TPESampler(seed=123, n_startup_trials=args.n_startup_trials, multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=args.n_startup_trials, n_warmup_steps=0, interval_steps=args.interval_steps)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def _obj(trial): return objective(trial, fixed)
    study.optimize(_obj,
                   n_trials=None,
                   timeout=args.timeout,
                   gc_after_trial=True,
                   callbacks=[lambda s,t: leaderboard_callback(s, t, out_dir=f"{fixed.local_root}/hpo_results")])


