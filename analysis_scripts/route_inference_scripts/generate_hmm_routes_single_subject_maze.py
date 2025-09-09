import argparse
from pathlib import Path
from lmdphelper.train_entry import run_once


if __name__ == "__main__":
    root_dir = Path(__file__).parents[2]
    ap = argparse.ArgumentParser()
    # data / split
    ap.add_argument('--dataset', type=str, default='lmdp_agents')
    ap.add_argument('--maze_number', type=int, default=2)
    ap.add_argument('--subject_index', type=int, default=2)
    ap.add_argument('--kfold', type=int, default=None)
    ap.add_argument('--sample_ratio', type=float, default=1)
    # model params
    ap.add_argument('--n_routes', type=int, default=5)
    ap.add_argument('--cognitive_constant', type=float, default=11.51)  # 11.51
    ap.add_argument('--action_cost', type=float, default=0.1625)
    ap.add_argument('--reward_value', type=float, default=1.07)  # 1.07
    ap.add_argument('--route_entropy_param', type=int, default=0.1)
    ap.add_argument('--action_entropy_param', type=int, default=0.2)
    ap.add_argument('--noise', type=float, default=0)
    ap.add_argument('--noise_decay', type=float, default=1)
    # training
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=1500)
    ap.add_argument('--eval_every', type=int, default=30)
    ap.add_argument('--smooth_k', type=int, default=5)
    ap.add_argument('--log_every', type=int, default=10)
    # io
    ap.add_argument('--saving', type=int, default=0)
    ap.add_argument('--tensorboard', type=int, default=0)
    ap.add_argument('--output_dir', type=str, default=f'{root_dir}/inferred_routes/lowrank_lmdp_inferred/temp')
    # ap.add_argument('--true_routes_dir', type=str, default=f'{root_dir}/inferred_routes/lowrank_lmdp_inferred/temp')  # point to your project root if needed
    args = ap.parse_args()
    test = run_once(args)



# import torch
# from sklearn.covariance import log_likelihood
# from torch.utils.tensorboard import SummaryWriter
# import torch.nn.functional as F
# import pandas as pd
#
# from lowrank_lmdp.model import LowRankLMDP_HMM
# from datahelper.load_data import get_data, load_subject_IDs
# from mazehelper.transition_matrix_functions import location_action_adjacency_matrix_from_maze_id
#
# import os
# import argparse
# from pathlib import Path
# import hashlib, json
# import ast
# import shutil
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Using device: {device}')
#
#
# if __name__ == "__main__":
#     # define arguments and directories
#     # -----------------------------------------------------------------------------------------------------------------
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, default='lmdp_agents')
#     parser.add_argument('--maze_number', type=int, default=2)
#     parser.add_argument('--subject_index', type=int, default=2)
#     parser.add_argument('--kfold', type=int, default=None)
#
#     # define relevant model parameters
#     # -----------------------------------------------------------------------------------------------------------------
#     parser.add_argument('--n_routes', type=int, default=5)
#     parser.add_argument('--cognitive_constant', type=float, default=11.51)
#     parser.add_argument('--action_cost', type=float, default=0.1625)
#     parser.add_argument('--reward_value', type=float, default=1.07)
#
#     parser.add_argument('--route_entropy_param', type=int, default=0)
#     parser.add_argument('--action_entropy_param', type=int, default=0)
#     parser.add_argument('--noise', type=float, default=0)
#     parser.add_argument('--noise_decay', type=float, default=1)
#
#     # define relevant training parameters
#     # -----------------------------------------------------------------------------------------------------------------
#     parser.add_argument('--lr', type=float, default=0.01)
#     parser.add_argument('--seed', type=int, default=0)
#     parser.add_argument('--epochs', type=int, default=10000)
#
#     # other parameters
#     # -----------------------------------------------------------------------------------------------------------------
#     parser.add_argument('--saving', type=int, default=0)
#
#     args, _ = parser.parse_known_args()
#     dataset = args.dataset
#     maze_number = args.maze_number
#     subject_ID = load_subject_IDs(dataset)[args.subject_index]
#     kfold = args.kfold
#     print(f'Running inference for dataset: {dataset}, maze number: {maze_number}, subject ID: {subject_ID}, kfold: {kfold}')
#
#     # set up directories for model saving
#     # -----------------------------------------------------------------------------------------------------------------
#     root_dir = Path(__file__).parents[2]
#     if args.saving == 1:
#         print('setting up directories...')
#         model_dir = f'{root_dir}/inferred_routes/lowrank_lmdp_inferred/{dataset}/{subject_ID}/Maze{maze_number}_kfold_{kfold}'
#         os.makedirs(model_dir, exist_ok=True)
#         params = {
#             'n_routes': args.n_routes,
#             'cognitive_constant': args.cognitive_constant,
#             'action_cost': args.action_cost,
#             'reward_value': args.reward_value,
#             'route_entropy_param': args.route_entropy_param,
#             'action_entropy_param': args.action_entropy_param,
#             'noise': args.noise,
#             'noise_decay': args.noise_decay,
#             'lr': args.lr,
#             'seed': args.seed,
#             'epochs': args.epochs,
#         }
#
#         param_str = json.dumps(params, sort_keys=True).encode('utf-8')
#         config_hash = hashlib.md5(param_str).hexdigest()[:8]
#
#         with open(f"{model_dir}/config_{config_hash}.json", "w") as f:
#             json.dump(params, f, indent=2)
#         model_name = f'model_{config_hash}.pt'
#         model_path = f'{model_dir}/{model_name}'
#     else:
#         model_dir = f'{root_dir}/temp'
#         os.makedirs(model_dir, exist_ok=True)
#         model_path = f'{model_dir}/temp_model.pt'
#     print(f'Models will be saved to: {model_path}')
#
#     # check with real routes if available
#     # -----------------------------------------------------------------------------------------------------------------
#     if dataset == 'lmdp_agents' or dataset == 'dijkstra_agents':
#         routes_df = pd.read_csv(f'{root_dir}/data/synthetic_data/{dataset[:-7]}/{dataset}_meta_data_{maze_number}.csv')
#         routes_df['routes'] = routes_df['routes'].apply(ast.literal_eval)
#         real_routes = routes_df.loc[routes_df.subject_ID == subject_ID, 'routes'].iloc[0]
#         real_routes = torch.tensor(real_routes, dtype=torch.float32).to(device)
#
#     # saving file and the model used for reproducibility
#     # -----------------------------------------------------------------------------------------------------------------
#     if args.saving == 1:
#         shutil.copy(__file__, f'{model_dir}/script_{config_hash}.py')
#         shutil.copy(f'{root_dir}/lowrank_lmdp/model.py', f'{model_dir}/model_script_{config_hash}.py')
#
#     # set up training and validation data
#     # -----------------------------------------------------------------------------------------------------------------
#     torch.manual_seed(args.seed)
#     print('loading data...')
#     data = get_data(dataset=dataset, maze_number=maze_number, query={"subject_ID": subject_ID})
#     x = torch.tensor(data['pos_idx'].to_numpy(), dtype=torch.long).to(device)
#     a = torch.tensor(data['action_class'].to_numpy(), dtype=torch.long).to(device)
#     r = torch.tensor(data['reward_idx'].to_numpy(), dtype=torch.long).to(device)
#
#     if kfold is None:
#         _, trials_per_time = torch.tensor(data.trial.to_numpy()).to(device).unique_consecutive(
#             return_inverse=True)
#         trials = trials_per_time.unique()  # this is to ensure that trials are unique numbers across sessions
#         trials_perm = torch.randperm(trials.size(0)).to(device)
#         train_trials, validate_trials = trials_perm[:round(trials.size(0) * 0.85)], trials_perm[round(trials.size(0) * 0.85):]
#         x_t, a_t, r_t = x[torch.isin(trials_per_time, train_trials)].to(device), a[torch.isin(trials_per_time, train_trials)].to(device), r[torch.isin(trials_per_time, train_trials)].to(device)
#         x_v, a_v, r_v = x[torch.isin(trials_per_time, validate_trials)].to(device), a[torch.isin(trials_per_time, validate_trials)].to(device), r[torch.isin(trials_per_time, validate_trials)].to(device)
#     else:
#         filter = data.day_on_maze.isin([args.kfold + 1, args.kfold + 2])
#         x_t, a_t, r_t = x[~filter], a[~filter], r[~filter]
#         x_v, a_v, r_v = x[filter], a[filter], r[filter]
#
#
#     # define and load model
#     # -----------------------------------------------------------------------------------------------------------------
#     print('defining model...')
#     model = LowRankLMDP_HMM(
#         n_routes=args.n_routes,
#         n_locs=49,
#         n_acts_per_loc=4,
#         cognitive_constant=args.cognitive_constant,
#         action_cost=args.action_cost,
#         reward_value=args.reward_value,
#         route_entropy_param=args.route_entropy_param,
#         action_entropy_param=args.action_entropy_param,
#         noise=args.noise,
#         noise_decay=args.noise_decay,
#         adjacency_matrix=location_action_adjacency_matrix_from_maze_id(maze_id=args.maze_number),
#         lr=args.lr,
#     ).to(device)
#
#     # train model
#     # ---------------------------------------------------------------------------------------------------------
#     print('training model...')
#     validation_loss = -1e10
#
#     os.makedirs(f'{model_dir}/logs', exist_ok=True)
#     writer = SummaryWriter(log_dir=f'{model_dir}/logs')
#     for i in range(args.epochs):
#         log_prob, log_start, entropy_action_loss, entropy_route_loss, log_likelihood = model(x_t, a_t, r_t)
#         writer.add_scalar("log_likelihood/train", log_likelihood, i)
#         writer.add_scalar("log_transitions/train", log_prob, i)
#         writer.add_scalar("log_start/train", log_start, i)
#         writer.add_scalar("entropy_action_loss/train", entropy_action_loss, i)
#         writer.add_scalar("entropy_route_loss/train", entropy_route_loss, i)
#         if dataset == 'lmdp_agents' or dataset == 'dijkstra_agents':
#             corr = model.get_corr(real_routes=real_routes)
#             writer.add_scalar("correlation/train", corr, i)
#
#         # validation every 100 steps
#         if i % 100 == 0 or i == args.epochs - 1:
#             with torch.no_grad():
#                 log_prob_v, log_start_v = model.E_step(x_v, a_v, r_v)
#             writer.add_scalar("log_transitions/validate", log_prob_v, i)
#             writer.add_scalar("log_start/validate", log_start_v, i)
#             writer.add_scalar("log_likelihood/validate", log_prob_v + log_start_v, i)
#             writer.flush()
#             if (log_prob_v + log_start_v) > validation_loss:
#                 validation_loss = log_prob_v + log_start_v
#                 print(f'save {i}th iteration model')
#                 torch.save(model.state_dict(), model_path)
#     writer.close()
