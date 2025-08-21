import torch


class DijkstraRoutePlanner:
    def __init__(self,
                 n_locs: int,
                 n_acts_per_loc: int,
                 cognitive_constant: float,
                 action_cost: float,
                 reward_value: float,
                 adjacency_matrix: torch.Tensor,
                 routes: torch.Tensor,
                 prior: torch.Tensor = None,
                 ):
        """
        Initializes the DijkstraRoutePlanner with the given parameters.
        :param n_locs:
        :type n_locs:
        :param n_acts_per_loc:
        :type n_acts_per_loc:
        :param cognitive_constant:
        :type cognitive_constant:
        :param action_cost:
        :type action_cost:
        :param reward_value:
        :type reward_value:
        :param adjacency_matrix:
        :type adjacency_matrix:
        :param routes:
        :type routes:
        :param prior:
        :type prior:
        """
