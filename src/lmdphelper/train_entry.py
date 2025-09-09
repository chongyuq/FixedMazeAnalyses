# train_entry.py
import os, json, hashlib, ast, shutil
from pathlib import Path
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from lowrank_lmdp.model import LowRankLMDP_HMM
from datahelper.load_data import get_data, load_subject_IDs
from mazehelper.transition_matrix_functions import location_action_adjacency_matrix_from_maze_id

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def moving_avg(vals, k):
    if not vals: return float("-inf")
    k = min(k, len(vals))
    return sum(vals[-k:]) / k


def run_once(args, report_cb=None) -> dict:
    # ----- resolve dirs
    subject_id = load_subject_IDs(args.dataset)[args.subject_index]
    kfold_suffix = f"_kfold_{args.kfold}" if args.kfold is not None else ""

    if args.saving or args.tensorboard:
        out_root = (
                Path(args.output_dir).resolve()
                / args.dataset
                / str(subject_id)
                / f"Maze{args.maze_number}{kfold_suffix}"
        )
        out_root.mkdir(parents=True, exist_ok=True)

    # ----- data ids
    subject_ID = load_subject_IDs(args.dataset)[args.subject_index]
    # ----- seed
    torch.manual_seed(args.seed)

    # ----- data
    data = get_data(dataset=args.dataset, maze_number=args.maze_number, query={"subject_ID": subject_ID, 'trial_phase': "navigation"})
    data = data[data.pos_idx != data.reward_idx]  # remove all reward positions
    x = torch.tensor(data['pos_idx'].to_numpy(), dtype=torch.long).to(device)
    a = torch.tensor(data['action_class'].to_numpy(), dtype=torch.long).to(device)
    r = torch.tensor(data['reward_idx'].to_numpy(), dtype=torch.long).to(device)
    tpt = torch.tensor(data.trial.to_numpy(), dtype=torch.long).to(device)
    trial_frac = args.sample_ratio

    if args.kfold is None:
        _, trials_per_time = torch.tensor(data.trial.to_numpy()).to(device).unique_consecutive(return_inverse=True)
        trials = trials_per_time.unique()
        trials_perm = torch.randperm(trials.size(0)).to(device)
        split = round(trials.size(0) * 0.8)
        train_trials, validate_trials = trials_perm[:split], trials_perm[split:]
        mask_t = torch.isin(trials_per_time, train_trials)
        mask_v = torch.isin(trials_per_time, validate_trials)
    else:
        # simple fold example: hold out two consecutive days based on kfold index
        mask = data.day_on_maze.isin([args.kfold + 1, args.kfold + 2])
        mask_t, mask_v = ~torch.tensor(mask.values, device=device), torch.tensor(mask.values, device=device)

    # subsample a proportion of trials within each split
    if trial_frac < 1.0:
        def sub(mask):
            ids = torch.unique(tpt[mask])
            k = max(1, int(ids.numel() * trial_frac))
            keep = ids[torch.randperm(ids.numel(), device=device)[:k]]
            return mask & torch.isin(tpt, keep)

        mask_t, mask_v = sub(mask_t), sub(mask_v)

    x_t, a_t, r_t = x[mask_t], a[mask_t], r[mask_t]
    x_v, a_v, r_v = x[mask_v], a[mask_v], r[mask_v]

    # ----- real routes (optional for synthetic datasets)
    real_routes = None
    if args.dataset in ('lmdp_agents', 'dijkstra_agents'):
        root_dir = Path(__file__).parents[2]
        routes_df = pd.read_csv(f'{root_dir}/data/synthetic_data/{args.dataset[:-7]}/{args.dataset}_meta_data_{args.maze_number}.csv')
        routes_df['routes'] = routes_df['routes'].apply(ast.literal_eval)
        real_routes = routes_df.loc[routes_df.subject_ID == subject_ID, 'routes'].iloc[0]
        real_routes = torch.tensor(real_routes, dtype=torch.float32).to(device)

    # ----- model
    model = LowRankLMDP_HMM(
        n_routes=args.n_routes,
        n_locs=49,
        n_acts_per_loc=4,
        cognitive_constant=args.cognitive_constant,
        action_cost=args.action_cost,
        reward_value=args.reward_value,
        route_entropy_param=args.route_entropy_param,
        action_entropy_param=args.action_entropy_param,
        noise=args.noise,
        noise_decay=args.noise_decay,
        adjacency_matrix=location_action_adjacency_matrix_from_maze_id(maze_id=args.maze_number),
        lr=args.lr,
    ).to(device)

    model.R = torch.nn.Parameter(
        torch.logit(F.normalize(real_routes*(1-1e-12) + 1e-12*(1-real_routes), p=1, dim=-1)).to(device)
    )
    model.pi = torch.nn.Parameter(torch.logit(torch.ones(args.n_routes + 1).to(device) / (args.n_routes + 1))).to(device)
    model.calculate_transition_matrix_and_policies()

    # ----- params + hash
    if args.saving:
        params = dict(
            dataset=args.dataset, maze_number=args.maze_number, subject_index=args.subject_index, kfold=args.kfold,
            n_routes=args.n_routes, cognitive_constant=args.cognitive_constant, action_cost=args.action_cost,
            reward_value=args.reward_value, route_entropy_param=args.route_entropy_param, action_entropy_param=args.action_entropy_param,
            noise=args.noise, noise_decay=args.noise_decay, lr=args.lr, seed=args.seed, epochs=args.epochs,
            eval_every=args.eval_every, smooth_k=args.smooth_k
        )
        config_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
        best_path = out_root / f"best_{config_hash}.pt"
        with open(f"{out_root}/config_{config_hash}.json", "w") as f:
            json.dump(params, f, indent=2)

    # ----- logging
    if args.tensorboard:
        tb_dir = out_root / "logs" / f"{config_hash}"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
    else:
        writer = None

    # ----- training loop
    best_val = float("-inf")
    val_hist = []
    best_epoch = -1
    for i in range(args.epochs):
        log_prob, log_start, ent_a, ent_r, ll, acc = model(x_t, a_t, r_t)

        if writer and (i % args.log_every == 0):
            writer.add_scalar("train/log_likelihood", ll, i)
            writer.add_scalar("train/log_transitions", log_prob, i)
            writer.add_scalar("train/log_start", log_start, i)
            writer.add_scalar("train/entropy_action_loss", ent_a, i)
            writer.add_scalar("train/entropy_route_loss", ent_r, i)
            writer.add_scalar("train/accuracy", acc, i)

        if (i % args.eval_every == 0) or (i == args.epochs - 1):
            with torch.no_grad():
                log_prob_v, log_start_v, accuracy_v = model.E_step(x_v, a_v, r_v)
            # val_metric = float((log_prob_v + log_start_v).item())
            val_metric = float(accuracy_v.item())
            val_hist.append(val_metric)
            smoothed = moving_avg(val_hist, args.smooth_k)
            if report_cb is not None:
                report_cb(step=i, value=float(smoothed))

            if writer:
                writer.add_scalar("val/log_transitions", log_prob_v, i)
                writer.add_scalar("val/log_start", log_start_v, i)
                writer.add_scalar("val/metric_raw", val_metric, i)
                writer.add_scalar("val/metric_smoothed", smoothed, i)
                if real_routes is not None:
                    corr = model.get_corr(real_routes=real_routes)
                    writer.add_scalar("val/correlation", corr, i)

            if smoothed > best_val:
                best_val = smoothed
                best_epoch = i
                if args.saving:
                    torch.save(model.state_dict(), best_path)

    if writer:
        # choose only simple types; cast everything to plain Python scalars/strings
        hparams = {
            "dataset": str(args.dataset),
            "maze_number": int(args.maze_number),
            "subject_index": int(args.subject_index),
            "kfold": -1 if args.kfold is None else int(args.kfold),
            "sample_ratio": float(args.sample_ratio),
            "n_routes": int(args.n_routes),
            "cognitive_constant": float(args.cognitive_constant),
            "action_cost": float(args.action_cost),
            "reward_value": float(args.reward_value),
            "route_entropy_param": int(args.route_entropy_param),
            "action_entropy_param": int(args.action_entropy_param),
            "noise": float(args.noise),
            "noise_decay": float(args.noise_decay),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "epochs": int(args.epochs),
        }
        # at least one metric is required by the HParams plugin
        metrics = {
            "val/best_smoothed": float(best_val),
            "val/best_epoch": float(best_epoch),
        }
        # log once per run
        writer.add_hparams(hparams, metrics, run_name=f"hparams_{config_hash}")
        writer.flush()

    if writer:
        writer.close()

    # ----- final results
    if args.saving or args.tensorboard:
        result = {
            "val_metric": float(best_val),
            "best_epoch": int(best_epoch),
            "config_hash": config_hash,
            "best_checkpoint": str(best_path) if args.saving else "",
            "output_dir": str(out_root),
            "device": str(device),
        }
    else:
        result = {
            "val_metric": float(best_val),
            "best_epoch": int(best_epoch),
            "device": str(device),
        }
    # Save + print one final JSON line
    if args.saving:
        (out_root / f"result_{config_hash}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return result