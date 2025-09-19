import optuna
import os
from types import SimpleNamespace
from lmdphelper.train_entry import run_once


def objective(trial: optuna.Trial, fixed):
    # --- search space ---
    params = dict(
        dataset=fixed.dataset,
        maze_number=fixed.maze_number,
        subject_index=fixed.subject_index,
        kfold=fixed.kfold,
        sample_ratio=fixed.sample_ratio,

        n_routes= fixed.n_routes, #trial.suggest_int("n_routes", 5, 8),
        cognitive_constant=trial.suggest_float("cognitive_constant", 8, 32),
        action_cost=trial.suggest_float("action_cost", 0.07, 0.23),
        # reward_value=trial.suggest_float("reward_value", 1, 3),
        reward_value=fixed.reward_value,
        route_entropy_param=trial.suggest_float("route_entropy_param", 0.0, 0.2),
        action_entropy_param=fixed.action_entropy_param,
        # action_entropy_param=trial.suggest_float("action_entropy_param", 0.05, 0.5),
        noise=fixed.noise,
        noise_decay=fixed.noise_decay,

        lr = fixed.lr,
        # lr=trial.suggest_float("lr", 1e-4, 5e-2, log=True),
        eval_every=fixed.eval_every,
        smooth_k=fixed.smooth_k,
        log_every=fixed.log_every,

        saving=fixed.saving,
        tensorboard=fixed.tensorboard,
    )

    # Create a unique directory for this trial to store outputs
    local_root = fixed.local_root
    # local_root = "/Users/chongyuqin/PhD/fixed_maze_analyses/inferred_routes/lowrank_lmdp_inferred/temp"
    if fixed.saving:
        out_dir = f"{local_root}/trial_{trial.number:05d}"
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = None

    # --- run training ---
    def report_cb(step, value):
        trial.report(value, step)
        if step >= fixed.min_epochs and trial.should_prune():
            raise optuna.TrialPruned(f"Trial was pruned at step {step}. Value: {value:.5f}")

    ns = SimpleNamespace(**params, epochs=fixed.max_epochs, seed=trial.suggest_int("seed", 1, 10000), output_dir=out_dir)
    res = run_once(ns, report_cb=report_cb)
    return float(res["val_metric"])
