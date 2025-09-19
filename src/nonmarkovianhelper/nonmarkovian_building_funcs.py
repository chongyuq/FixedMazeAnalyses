from datahelper.load_data import load_optimal_behaviour, load_subject_IDs, get_data, load_all_kfold_habits_for_dataset, load_all_kfold_pcs_for_dataset
from datahelper.fields import COMMON_FIELDS

from mazehelper.mazes_info import POSSIBLE_LOCATION_ACTION_FOR_MAZE_ID

import torch
import torch.nn.functional as F
from collections import defaultdict, OrderedDict
from sklearn.utils import resample
import statsmodels.api as sm


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


class SingleActionUniquePredictabilityFinder:
    def __init__(self, regressors):
        self.regressors = regressors
        self.n_regressors = len(regressors) + 1  # +1 for the constant regressor
        return

    def bootstrap_or_regress_per_subject(self, subject_IDs: list, X_t: list, Y_t: list, X_v: list, Y_v: list, bootstraps: int = None) -> dict:
        """
        Performs bootstrapping or regression per subject to find unique predictability of the regressors.
        :param subject_IDs: List of subject IDs to bootstrap or regress over. Although this is called subject_ID, it can be any identifier for a group of data points as long as each index corresponds to the same index in X_t, Y_t, X_v, and Y_v.
        :type subject_IDs: list
        :param X_t: Training data features for regression training. It's a list of tensors, each tensor is of shape (n, n_actions, n_regressors + 2) where n is the number of observations, n_actions is 4, and n_regressors is the number of regressors including the constant.
        The list is indexed by subject_IDs.
        :type X_t: list of torch.Tensor
        :param Y_t: Training data labels for regression training. It's a list of tensors, each tensor is of shape (n, n_actions) where n is the number of observations and n_actions is 4.
        The list is indexed by subject_IDs.
        :type Y_t: list of torch.Tensor
        :param X_v: Validation data features for regression validation. It's a list of tensors, each tensor is of shape (n, n_actions, n_regressors + 2) where n is the number of observations, n_actions is 4, and n_regressors is the number of regressors including the constant.
        The list is indexed by subject_IDs.
        :type X_v: list of torch.Tensor
        :param Y_v: Validation data labels for regression validation. It's a list of tensors, each tensor is of shape (n, n_actions) where n is the number of observations and n_actions is 4.
        The list is indexed by subject_IDs.
        :type Y_v: list of torch.Tensor
        :param bootstraps: How many bootstraps to perform. If None, it will bootstrap over all subjects.
        :type bootstraps: int or None
        :return: A dictionary with keys 'coefs', 'neg_log_likelihood', and 'accuracies'.
        'coefs' is a tensor of shape (bootstraps, n_regressors + 1, n_regressors) where the first row corresponds to the constant regressor and the rest correspond to the regressors in the order they were provided.
        The last row corresponds to the full model with all regressors.
        'neg_log_likelihood' is a tensor of shape (bootstraps, n_regressors + 1) where each element corresponds to the log likelihood of the validation data given the coefficients.
        'accuracies' is a tensor of shape (bootstraps, n_regressors + 1) where each element corresponds to the accuracy of the validation data given the coefficients.
        :rtype: dictionary
        """
        assert len(subject_IDs) == len(X_t) == len(Y_t) == len(X_v) == len(Y_v)
        bootstrapping = bootstraps is not None
        bootstraps = bootstraps or len(subject_IDs)
        random_states = (
            torch.randint(0, 2 ** 32 - 1, (bootstraps,))
            if bootstrapping else None
        )

        # Initialize tensors to store coefficients, neg_log_likelihood, and accuracies
        coefs = torch.zeros(bootstraps, self.n_regressors + 1, self.n_regressors)
        neg_log_likelihoods = torch.zeros(bootstraps, self.n_regressors + 1)
        accuracies = torch.zeros(bootstraps, self.n_regressors + 1)

        for bootstrap in range(bootstraps):
            if bootstrapping:
                subjects_sampled = resample(subject_IDs, random_state=random_states[bootstrap].item())
            else:
                subjects_sampled = [subject_IDs[bootstrap]]

            X_bootstrap_t = torch.cat([X_t[subject_IDs.index(subject)] for subject in subjects_sampled], dim=0)  # shape: (n, 4, n_regressors + 2)
            X_bootstrap_v = torch.cat([X_v[subject_IDs.index(subject)] for subject in subjects_sampled], dim=0)  # shape: (n, 4, n_regressors + 2)

            # Split the mask from the features
            # The last dimension is the mask for impossible actions, which is 0 for possible actions and -1e10 for impossible actions.
            # The mask is used to filter out impossible actions during regression. This step is required for get_unique_predictability to work correctly.
            # ------------------------------------------------------------------------------------------------------------
            X_bootstrap_t, M_bootstrap_t = torch.split(X_bootstrap_t, [X_bootstrap_t.size(-1) - 1, 1], dim=-1)
            X_bootstrap_v, M_bootstrap_v = torch.split(X_bootstrap_v, [X_bootstrap_v.size(-1) - 1, 1], dim=-1)
            M_bootstrap_t = (M_bootstrap_t == 0).squeeze(-1)  # 1 for possible actions, 0 for impossible actions # shape: (n, 4, 1)
            M_bootstrap_v = (M_bootstrap_v == 0).squeeze(-1)
            X_bootstrap_t = X_bootstrap_t[M_bootstrap_t]
            X_bootstrap_v = X_bootstrap_v[M_bootstrap_v]

            # Normalize the features and maintain the constant term
            # ------------------------------------------------------------------------------------------------------------
            X_bootstrap_t = (X_bootstrap_t - X_bootstrap_t.mean(dim=0, keepdim=True)) / (X_bootstrap_t.std(dim=0, keepdim=True) + 1e-11)
            X_bootstrap_t[..., 0] = 1  # constant term

            X_bootstrap_v = (X_bootstrap_v - X_bootstrap_v.mean(dim=0, keepdim=True)) / (X_bootstrap_v.std(dim=0, keepdim=True) + 1e-11)
            X_bootstrap_v[..., 0] = 1  # constant term

            # Get the actions for the training and validation data
            # ------------------------------------------------------------------------------------------------------------
            Y_bootstrap_t = torch.cat([Y_t[subject_IDs.index(subject)] for subject in subjects_sampled], dim=0)[M_bootstrap_t] # shape: (sum of M_bootstrap_t, )
            Y_bootstrap_v = torch.cat([Y_v[subject_IDs.index(subject)] for subject in subjects_sampled], dim=0)[M_bootstrap_v]

            # Get the unique predictability of the regressors
            # ------------------------------------------------------------------------------------------------------------
            unique_predictability = self.get_unique_predictability(X_t=X_bootstrap_t, Y_t=Y_bootstrap_t, M_t=M_bootstrap_t,
                                                                   X_v=X_bootstrap_v, Y_v=Y_bootstrap_v, M_v=M_bootstrap_v)
            coefs[bootstrap] = unique_predictability['coefs']
            neg_log_likelihoods[bootstrap] = unique_predictability['neg_log_likelihoods']
            accuracies[bootstrap] = unique_predictability['accuracies']
        return {
            'coefs': coefs,
            'neg_log_likelihoods': neg_log_likelihoods,
            'accuracies': accuracies
        }

    def get_unique_predictability(self, X_t, Y_t, M_t, X_v, Y_v, M_v):
        """
        Calculates the unique predictability of the regressors by optimizing models with missing regressors and comparing it to the full model.
        :param X_t: Training data features for regression training. Shape (n, n_actions, n_regressors + 1) where n is the number of observations, n_actions is 4, and n_regressors is the number of regressors including the constant,
        The last dimension is the mask for impossible actions.
        :type X_t: torch.Tensor
        :param Y_t: Training data labels for regression training. Shape (n, n_actions) where n is the number of observations and n_actions is 4.
        :type Y_t: torch.Tensor
        :param M_t: Mask for impossible actions in the training data. Shape (n, n_actions, 1) where n is the number of observations and n_actions is 4.
        :type M_t: torch.Tensor
        :param X_v: Validation data features for regression validation. Shape (n, n_actions, n_regressors + 1) where n is the number of observations, n_actions is 4, and n_regressors is the number of regressors including the constant.
        The last dimension is the mask for impossible actions.
        :type X_v: torch.Tensor
        :param Y_v: Validation data labels for regression validation. Shape (n, n_actions) where n is the number of observations and n_actions is 4.
        :type Y_v: torch.Tensor
        :param M_v: Mask for impossible actions in the validation data. Shape (n, n_actions, 1) where n is the number of observations and n_actions is 4.
        :type M_v: torch.Tensor
        :return: A dictionary with keys 'coefs', 'neg_log_likelihood', and 'accuracies'.
        'coefs' is a tensor of shape (n_regressors + 1, n_regressors) where the first row corresponds to the constant regressor and the rest correspond to the regressors in the order they were provided.
        The last row corresponds to the full model with all regressors.
        'neg_log_likelihood' is a tensor of shape (n_regressors + 1) where each element corresponds to the log likelihood of the validation data given the coefficients.
        'accuracies' is a tensor of shape (n_regressors + 1) where each element corresponds to the accuracy of the validation data given the coefficients.
        :rtype: dict
        """
        # coefficients for the regressors
        # ------------------------------------------------------------------------------------------------------------
        # the coefficients are of shape (n_regressors + 1, n_regressors)
        # where the first row corresponds to annealing the constant regressor
        # and the rest correspond to the regressors in the order they were provided
        # the final row corresponds to the full model with all regressors
        # the coefficients are optimized using scipy.optimize.minimize
        coefs = torch.zeros(self.n_regressors + 1, self.n_regressors)
        neg_log_likelihoods= torch.zeros(self.n_regressors + 1)
        accuracies = torch.zeros(self.n_regressors + 1)
        for anneal_index in range(self.n_regressors + 1):
            # anneal over the regressors
            # also need to have a full model, so one more model than self.n_regressors -1, altogether self.n_regressors models
            # ------------------------------------------------------------------------------------------------------------
            if anneal_index == self.n_regressors:
                indices = list(range(self.n_regressors))
            else:
                indices = list(range(anneal_index)) + list(range(anneal_index + 1, self.n_regressors))
            est = sm.Logit(Y_t.int().numpy(), X_t[..., indices].numpy()).fit()
            coefs[anneal_index, indices] = torch.tensor(est.params).float()

            # validation
            # loss is the log likelihood of the validation data given the coefficients
            # accuracy is the accuracy of the validation data given the coefficients
            # ------------------------------------------------------------------------------------------------------------
            neg_log_likelihood = -sm.Logit(Y_v.int().numpy(), X_v[..., indices].numpy()).loglikeobs(est.params).mean()
            accuracy = ((Y_v * torch.tensor(est.predict(X_v[..., indices].numpy())).float()).sum(dim=-1) > 0.5).float().mean()
            neg_log_likelihoods[anneal_index] = torch.tensor(neg_log_likelihood)
            accuracies[anneal_index] = torch.tensor(accuracy)
        return {
            'coefs': coefs,
            'neg_log_likelihoods': neg_log_likelihoods,
            'accuracies': accuracies
        }



