from datahelper.load_data import load_optimal_behaviour, load_subject_IDs, get_data
from datahelper.fields import COMMON_FIELDS

from mazehelper.mazes_info import POSSIBLE_LOCATION_ACTION_FOR_MAZE_ID
from mazehelper.transition_matrix_functions import location_action_adjacency_matrix_from_maze_id

from regressionhelper.constants import *
from regressionhelper.regressor_building_funcs import log_likelihood_with_fixed_constant, accuracy_with_fixed_constant

from lowrank_lmdp.model import LowRankLMDP_HMM

import numpy as np
import torch
import torch.functional as F
from sklearn.utils import resample
from functools import partial
from scipy.optimize import minimize


class TrainingDataBuilder:
    def __init__(self,
                 dataset: str,
                 maze_number: int,
                 regressors: list,
                 bootstrap: bool = False):
        """
        Initializes the TrainingDataBuilder with the given parameters.
        :param dataset: The dataset identifier.
        The possible values are 'mice_behaviour', 'lmdp_agents', 'dijkstra_agents', 'non_markovian_agents', 'markovian_agents'.
        :type dataset: str
        :param maze_number: The maze number to use for the data.
        :type maze_number: int
        :param regressors: List of regressors to use for the data.
        :type regressors: list
        :param bootstrap: If True, bootstrap the data across subjects. Otherwise, regress individual subjects
        :type bootstrap: bool
        """

        assert dataset in COMMON_FIELDS, f"Dataset {dataset} is not recognized. Available datasets: {COMMON_FIELDS.keys()}."
        assert maze_number in [1, 2, 3], "Maze number must be 1, 2, or 3."
        assert [regressor in REGRESSORS for regressor in regressors], \
            f"Regressors must be in {REGRESSORS}. Provided: {regressors}"

        self.dataset = dataset
        self.maze_number = maze_number
        self.regressors = regressors
        self.subject_IDs = load_subject_IDs(dataset)
        self.bootstraps = len(self.subject_IDs)
        self.optimal, self.habit, self.hmm_routes, self.pca_routes = None, None, None, None
        self.hmm_route_state_dict, self.pca_state_dict = None, None

        if 'optimal' in regressors:
            self.optimal = load_optimal_behaviour(maze_number)
        # if 'habit' in regressors:
        #     self.habit = load_habit(maze_number, dataset=dataset)
        # if 'hmm_route' or 'hmm_route_planning' in regressors:
        #     self.hmm_route_state_dict = load_hmm_route_state_dicts(maze_number, dataset=dataset, folds=True)
        # if 'PCA_route' or 'PCA_route_planning' in regressors:
        #     self.pca_routes = load_pca_routes(maze_number, dataset=dataset, folds=True)
        self.n_sessions = get_data(
            agent_type=dataset,
            maze_number=maze_number,
        ).day_on_maze.nunique()

    def build(self):
        X_t, Y_t, X_v, Y_v = [], [], [], []
        for i, subject in enumerate(self.subject_IDs):
            processor = SubjectProcessor(
                dataset=self.dataset,
                maze_number=self.maze_number,
                subject_ID=subject,
            )
            for k_fold in range(self.n_sessions - 1):
                subject_data = processor.process(k_fold)
                X_t.append(subject_data['X_t'])
                Y_t.append(subject_data['Y_t'])
                X_v.append(subject_data['X_v'])
                Y_v.append(subject_data['Y_v'])
        return X_t, Y_t, X_v, Y_v



class UniquePredictabilityFinder:
    def __init__(self, regressors):
        self.regressors = regressors
        self.n_regressors = len(regressors) + 1  # +1 for the constant regressor
        return

    def get_unique_predictability(self, X_t, Y_t, X_v, Y_v):
        """
        Calculates the unique predictability of the regressors by optimizing models with missing regressors and comparing it to the full model.
        :param X_t:
        :type X_t:
        :param Y_t:
        :type Y_t:
        :param X_v:
        :type X_v:
        :param Y_v:
        :type Y_v:
        :return:
        :rtype:
        """
        X_t, C_t = torch.split(X_t, [X_t.size(-1) - 1, 1], dim=-1)
        X_v, C_v = torch.split(X_v, [X_v.size(-1) - 1, 1], dim=-1)

        coefs = torch.zeros(self.n_regressors + 1, self.n_regressors)
        losses = torch.zeros(self.n_regressors + 1)
        accuracies = torch.zeros(self.n_regressors + 1)
        for anneal_index in range(self.n_regressors + 1):
            # anneal over the regressors
            # also need to have a full model, so one more model than self.n_regressors -1, altogether self.n_regressors models
            # ------------------------------------------------------------------------------------------------------------
            if anneal_index == self.n_regressors:
                indices = list(range(self.n_regressors))
            else:
                indices = list(range(anneal_index)) + list(range(anneal_index + 1, self.n_regressors))
            g = partial(log_likelihood_with_fixed_constant, X_t[..., indices].detach().numpy(), Y_t.max(dim=-1)[1].detach().numpy(), C_t.squeeze().detach().numpy())
            w = minimize(g, np.random.random(len(indices)) * 0.001)
            coefs[anneal_index, indices] = torch.tensor(w.x).float()

            # validation
            # ------------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------------
            loss = log_likelihood_with_fixed_constant(X_v[..., indices].detach().numpy(), Y_v.max(dim=-1)[1].detach().numpy(), C_v.squeeze().detach().numpy(), w.x)
            accuracy = accuracy_with_fixed_constant(X_v[..., indices].detach().numpy(), Y_v.max(dim=-1)[1].detach().numpy(), C_v.squeeze().detach().numpy(), w.x)
            losses[anneal_index] = torch.tensor(loss)
            accuracies[anneal_index] = torch.tensor(accuracy)
        return


class SubjectProcessor:
    def __init__(self,
                 dataset,
                 maze_number,
                 subject_ID,
                 pca_state_dict=None,
                 hmm_state_dict=None,
                 habit=None,
                 optimal=None,
                 regressors=[]):
        """
        Initializes the SubjectProcessor with the given parameters.
        This class is responsible for processing the data for a specific subject and maze number.
        It prepares the data for regression by extracting relevant features and applying necessary transformations.
        The regressors accepted is defined regressorhelper.constants REGRESSORS.
        The dataset accepted are the keys in datahelper.fields COMMON_FIELDS.
        :param dataset: The dataset identifier, e.g., 'mice_behaviour', 'lmdp_agents', etc.
        :type dataset: str
        :param maze_number: The maze number to use for the data, should be 1, 2, or 3.
        :type maze_number: int
        :param subject_ID: The unique identifier for the subject. 'm2', 'm3', etc. for mice, 'lmdp1', 'lmdp2', etc. for LMDP agents.
        :type subject_ID: str
        :param pca_state_dict: The state dictionary for the PCA model, if available.
        :type pca_state_dict: dict, optional
        :param hmm_state_dict: The state dictionary for the HMM model, if available.
        :type hmm_state_dict: dict, optional
        :param habit: The habit data for the subject, if available. Should be a tensor of shape (n_folds, 49, 4) where n_folds is the number of folds (or days - 1).
        :type habit: torch.Tensor, optional
        :param optimal: The optimal behaviour data for the maze, if available. Should be a tensor of shape (49, 49, 4) where the first dimension is the reward index,
        the second dimension is the location index, and the third dimension is the action index.
        :type optimal: torch.Tensor, optional
        :param regressors: List of regressors to use for the regression. Should be in the list defined in regressionhelper.constants REGRESSORS.
        :type regressors: list, optional
        """
        assert dataset in COMMON_FIELDS, f"Dataset {dataset} is not recognized."
        assert maze_number in [1, 2, 3], "Maze number must be 1, 2, or 3."
        assert regressors in REGRESSORS, \
            f"Regressors must be in {REGRESSORS}. Provided: {regressors}"
        self.dataset = dataset
        self.maze_number = maze_number
        self.subject_id = subject_ID
        self.pca_state_dict = pca_state_dict
        self.hmm_state_dict = hmm_state_dict
        self.habit = habit  # shape n_folds (or n_days -1) x 49 x 4
        self.optimal = optimal  # shape: 49 x 49 x 4, or reward x location x action
        self.regressors = regressors

    def process(self, kfold):
        """
        Processes the data for a specific subject and k-fold.
        :param kfold: kfold refers to a split of the data where 'day_on_maze' is kfold + 1 is used for training regression and 'day_on_maze' is kfold + 2 is used for validation.
        :type kfold: int, should be between 0 and the total number of days on maze - 1
        :return: a dictionary with keys 'X_t', 'Y_t', 'X_v', 'Y_v' where X_t and Y_t are training data and X_v and Y_v are validation data.
        X_t and X_v are torch.Tensors of shape (n, n_actions, n_regressors) where n is the number of observations, n_actions is 4, and n_regressors is the number of regressors + 1 (for the constant regressor).
        Y_t and Y_v are torch.Tensors of shape (n, n_actions) where n is the number of observations and n_actions is 4.
        :rtype: dict
        """
        self.init_model(kfold)
        maze_behaviour = get_data(
            dataset=self.dataset,
            maze_number=self.maze_number,
            query={
                 'subject_ID': self.subject_ID,
                 '$or': [{'day_on_maze': kfold + 1},
                         {'day_on_maze': kfold + 2}]
             })

        # -------------------------------------------------------------------------------------------
        maze_behaviour['sa'] = maze_behaviour['pos_idx'] + maze_behaviour['action_class'] * 49
        maze_behaviour['prev_pos_idx'] = maze_behaviour['pos_idx'].shift(1)
        maze_behaviour['previous_action'] = maze_behaviour['action_class'].shift(1)
        maze_behaviour['reverse_previous_action'] = (maze_behaviour['previous_action'] - 2) % 4
        x = torch.tensor(maze_behaviour.pos_idx.to_numpy()).long()
        a = torch.tensor(maze_behaviour.action_class.to_numpy()).long()
        r = torch.tensor(maze_behaviour.reward_idx.to_numpy()).long()
        x_p = torch.tensor(maze_behaviour.prev_pos_idx.to_numpy()).long()
        a_p = torch.tensor(maze_behaviour.previous_action.to_numpy()).long()

        # -------------------------------------------------------------------------------------------
        self.get_route_and_route_planning_regressor(x, a, r, x_p, a_p)
        if 'vector' in self.regressors:
            self.vector_regressor = torch.tensor([maze_behaviour.reward_cos_angle, maze_behaviour.reward_sin_angle, -maze_behaviour.reward_cos_angle, -maze_behaviour.reward_sin_angle]).t()  # shape: n x 4
        if 'optimal' in self.regressors:
            self.optimal_regressor = torch.tensor(self.optimal[maze_behaviour.reward_idx, maze_behaviour.pos_idx])  # shape: n x 4
        if 'habit' in self.regressors:
            self.habit_regressor = self.habit[kfold, x]
        if 'forward' in self.regressors:
            self.forward_regressor = torch.cat([torch.zeros(1, 4), F.one_hot(a_p[1:].long(), 4)])
        if 'reverse' in self.regressors:
            self.reverse_regressor = torch.cat(
                [torch.zeros(1, 4), F.one_hot(torch.tensor(maze_behaviour.reverse_previous_action)[1:].long(), 4)])

        Y = F.one_hot(a, 4) # shape: n x 4
        regressors = [torch.ones(x.shape[0])]  # bias regressor
        for regressor in self.regressors:
            regressors.append(getattr(self, f"{regressor}_regressor"))
        X = torch.stack(regressors, dim=-1) # shape: n x 4 x n_regressors
        M = 1 - torch.tensor(POSSIBLE_LOCATION_ACTION_FOR_MAZE_ID[self.maze_number]).reshape(4, 49).t()  # Mask for impossible actions, 1 for possible, 0 for impossible
        # shape is 49 x 4 where each row corresponds to a location and each column to an action

        # separate into a regression training set and a validation set
        # -----------------------------------------------------------
        filter_t = (maze_behaviour.trial_phase == 'navigation') & \
                   (maze_behaviour.reward_idx != maze_behaviour.pos_idx) & \
                   (maze_behaviour.day_on_maze == kfold + 1)
        filter_v = (maze_behaviour.trial_phase == 'navigation') & \
                   (maze_behaviour.reward_idx != maze_behaviour.pos_idx) & \
                   (maze_behaviour.day_on_maze == kfold + 2)
        X_t, Y_t = X[filter_t], Y[filter_t]
        X_v, Y_v = X[filter_v], Y[filter_v]
        mask_t, mask_v = M[x[filter_t]], M[x[filter_v]]

        # apply masking of impossible actions
        # -----------------------------------------------------------
        X_t = torch.cat([X_t, mask_t.masked_fill(mask_t==1, -1e10)[..., None]], dim=-1)
        X_v = torch.cat([X_v, mask_v.masked_fill(mask_v==1, -1e10)[..., None]], dim=-1)
        # note here the mask is added!!
        return {'X_t': X_t, 'Y_t': Y_t, 'X_v': X_v, 'Y_v': Y_v}

    def init_model(self, kfold):
        """ Initializes the PCA and HMM models if the state dictionaries are provided.
        :param kfold: kfold refers to a split of the data where 'day_on_maze' is kfold + 1 is used for training regression and 'day_on_maze' is kfold + 2
        is used for validation.
        :type kfold: int, should be between 0 and the total number of days on maze - 1
        :return: None, but initializes the PCA and HMM models and their routes if the state dictionaries are provided.
        :rtype: None
        """
        if self.pca_state_dict is not None:
            self.pca_model = LowRankLMDP_HMM(
                n_locs=49,
                n_acts_per_loc=4,
                n_routes=7,
                cognitive_constant=20,
                action_cost=0.15,
                reward_value=1,
                route_entropy_param=0,
                action_entropy_param=0,
                noise=0,
                noise_decay=0,
                adjacency_matrix=location_action_adjacency_matrix_from_maze_id(self.maze_number)
            )
            self.pca_model.load_state_dict(self.pca_state_dict[kfold])
            self.pca_model.init_params()
            self.pca_model.eval()
            self.pca_routes = self.pca_model.state_dist_given_route
        if self.hmm_state_dict is not None:
            self.hmm_model = LowRankLMDP_HMM(
                n_locs=49,
                n_acts_per_loc=4,
                n_routes=7,
                cognitive_constant=20,
                action_cost=0.15,
                reward_value=1,
                route_entropy_param=0,
                action_entropy_param=0,
                noise=0,
                noise_decay=0,
                adjacency_matrix=location_action_adjacency_matrix_from_maze_id(self.maze_number)
            )
            self.hmm_model.load_state_dict(self.hmm_state_dict[kfold])
            self.hmm_model.init_params()
            self.hmm_model.eval()
            self.hmm_routes = self.hmm_model.state_dist_given_route
        return

    def get_route_and_route_planning_regressor(self, x, a, r, x_p, a_p):
        """
        Calculates the route and route planning regressors based on the current state, action, and reward
        :param x: location indices tensor
        :param a: action indices tensor
        :param r: reward indices tensor
        :param x_p: previous location indices tensor
        :param a_p: previous action indices tensor
        :return: None, but updates the class attributes with the calculated regressors
        """
        if 'pca_route' in self.regressors or 'pca_route_planning' in self.regressors:
            pca_route_predict = self.pca_model._forward_algorithm_no_parallel(location=x, action=a, reward=r)
            pca_route_predict = F.normalize(pca_route_predict, dim=-1)
            # need to use the previous route prediction in the regression, as the current behavioural step was
            # used to find the route
            pca_route_predict = torch.cat([torch.zeros(1, pca_route_predict.shape[1]), pca_route_predict[:-1]], dim=0)
            self.pca_route_regressor = torch.einsum('bj, jab->ba', pca_route_predict, self.pca_routes.reshape(-1, 4, 49)[..., x])
            # here note we use the previous route distribution
            # with the previous location and action - which gives the current location
            # with the current reward to calculate the route planning regressor
            pca_route_planning_regressor = torch.einsum('bsaij, bi -> bsaj',
                                           self.pca_model.policy[r[1:], x_p[1:], a_p[1:]],
                                           pca_route_predict[1:]).sum(dim=(1, -1))
            self.pca_route_planning_regressor = torch.cat([torch.zeros_like(pca_route_planning_regressor[0:1]), pca_route_planning_regressor], dim=0)
        if 'hmm_route' in self.regressors or 'hmm_route_planning' in self.regressors:
            hmm_route_predict = self.hmm_model._forward_algorithm_no_parallel(location=x, action=a, reward=r)
            hmm_route_predict = F.normalize(hmm_route_predict, dim=-1)
            # need to use the previous route prediction in the regression, as the current behavioural step was
            # used to find the route
            hmm_route_predict = torch.cat([torch.zeros(1, hmm_route_predict.shape[1]), hmm_route_predict[:-1]], dim=0)
            self.hmm_route_regressor = torch.einsum('bj, jab->ba', hmm_route_predict, self.hmm_routes.reshape(-1, 4, 49)[..., x])
            # here note we use the previous route distribution
            # with the previous location and action - which gives the current location
            # with the current reward to calculate the route planning regressor
            hmm_route_planning_regressor = torch.einsum('bsaij, bi -> bsaj',
                                           self.hmm_model.policy[r[1:], x_p[1:], a_p[1:]],
                                           hmm_route_predict[1:]).sum(dim=(1, -1))
            self.hmm_route_planning_regressor = torch.cat([torch.zeros_like(hmm_route_planning_regressor[0:1]), hmm_route_planning_regressor], dim=0)
        return

