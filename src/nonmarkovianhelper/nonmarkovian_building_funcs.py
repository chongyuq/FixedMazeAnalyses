from datahelper.load_data import load_optimal_behaviour, load_subject_IDs, get_data, load_all_kfold_habits_for_dataset, load_all_kfold_pcs_for_dataset
from datahelper.fields import COMMON_FIELDS

from mazehelper.mazes_info import POSSIBLE_LOCATION_ACTION_FOR_MAZE_ID

import torch
import torch.nn.functional as F
from collections import defaultdict, OrderedDict


class NonMarkovianTrainingDataBuilder:
    def __init__(self,
                 dataset: str,
                 maze_number: int,
                 tot_steps_back: int = 4):
        """
        Initializes the TrainingDataBuilder with the given parameters.
        :param dataset: The dataset identifier.
        The possible values are 'mice_behaviour', 'lmdp_agents', 'dijkstra_agents', 'non_markovian_agents', 'markovian_agents'.
        :type dataset: str
        :param maze_number: The maze number to use for the data.
        :type maze_number: int
        :param tot_steps_back: Total number of steps back to consider for non-markovian features.
        """

        assert dataset in COMMON_FIELDS, f"Dataset {dataset} is not recognized. Available datasets: {COMMON_FIELDS.keys()}."
        assert maze_number in [1, 2, 3], "Maze number must be 1, 2, or 3."

        self.dataset = dataset
        self.maze_number = maze_number
        self.subject_IDs = load_subject_IDs(dataset)
        self.tot_steps_back = tot_steps_back
        self.data = get_data(dataset=dataset,
                             maze_number=maze_number,
                             query={
                                 'trial_phase': 'navigation',
                                 '$expr': {'$ne': ['$pos_idx', '$reward_idx']}
                             })
        self.n_sessions = self.data.day_on_maze.nunique()
        self.engineer_lagged_features()
        return

    def process(self, k_fold: int):
        T_training_data = self.data[~self.data['day_on_maze'].isin([k_fold + 1, k_fold + 2])]
        RV_data = self.data[self.data['day_on_maze'].isin([k_fold + 1, k_fold + 2])]

        transition_matrix = self.calculate_transition_matrix(T_training_data)

        # establish the regressors
        previous_action_regressors = torch.tensor(RV_data[[f'a_{t}'for t in range(1, self.tot_steps_back + 1)]].to_numpy()).long()
        previous_action_regressors = F.one_hot(previous_action_regressors, 4).float().permute(0, 2, 1) # shape: n x 4 x tot_steps_back
        transition_regressor = transition_matrix[RV_data['reward_idx'].to_numpy(), RV_data['pos_idx'].to_numpy()][..., None]  # shape: n x 1 x 4
        X = torch.cat([torch.ones(transition_regressor.shape[0], 4, 1), transition_regressor, previous_action_regressors], dim=-1)  # shape: n x 4 x (1 + 1 + tot_steps_back)

        # mask for impossible actions
        Y = F.one_hot(torch.tensor(RV_data['action_class'].to_numpy()).long(), 4) # shape: n x 4
        M = 1 - torch.tensor(POSSIBLE_LOCATION_ACTION_FOR_MAZE_ID[self.maze_number]).reshape(4, 49).t()  # Mask for impossible actions, 1 for possible, 0 for impossible
        # shape is 49 x 4 where each row corresponds to a location and each column to an action

        # separate into a regression training set and a validation set
        # -----------------------------------------------------------
        filter_t = (RV_data.day_on_maze == k_fold + 1).to_list()
        filter_v = (RV_data.day_on_maze == k_fold + 2).to_list()
        X_t, Y_t = X[filter_t], Y[filter_t]
        X_v, Y_v = X[filter_v], Y[filter_v]
        x = RV_data.pos_idx.to_numpy()
        mask_t, mask_v = M[x[filter_t]], M[x[filter_v]]

        # apply masking of impossible actions
        # -----------------------------------------------------------
        X_t = torch.cat([X_t, mask_t.masked_fill(mask_t==1, -1e10)[..., None]], dim=-1)
        X_v = torch.cat([X_v, mask_v.masked_fill(mask_v==1, -1e10)[..., None]], dim=-1)
        # note here the mask is added!!
        return {'X_t': X_t, 'Y_t': Y_t, 'X_v': X_v, 'Y_v': Y_v}

    def engineer_lagged_features(self):
        df = self.data.copy()
        group_keys = ["subject_ID", "maze_number", "day_on_maze", "trial"]
        # note that the df is sorted by subject_ID, day_on_maze, trial, time in get_data

        df["sa_0"] = df["action_class"] * 49 + df["pos_idx"]
        df["a_0"] = df["action_class"]

        for t in range(1, self.tot_steps_back + 1):
            df[f"sa_{t}"] = df.groupby(group_keys)["sa_0"].shift(t) # previous t state-action pair
            df[f"a_{t}"] = df.groupby(group_keys)["a_0"].shift(t) # previous t action

        self.data = df
        return

    def calculate_transition_matrix(self, data):
        # Calculate transition matrix from training data
        transition_matrix = torch.ones(49, 49, 4) * float('nan')
        for loc in range(49):
            transition_matrix[loc] = F.one_hot(torch.tensor(data[f'sa_0'][data[f'reward_idx']==loc].to_numpy()).long(), 196).float().mean(dim=0).reshape(4, 49).transpose(-2, -1)
        transition_matrix = F.normalize(transition_matrix, dim=-1)
        return transition_matrix

    def build(self):
        X_t, Y_t, X_v, Y_v = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        for k_fold in range(self.n_sessions - 1):
            data = self.process(k_fold)
            X_t[k_fold].append(data['X_t'])
            Y_t[k_fold].append(data['Y_t'])
            X_v[k_fold].append(data['X_v'])
            Y_v[k_fold].append(data['Y_v'])
        return X_t, Y_t, X_v, Y_v



