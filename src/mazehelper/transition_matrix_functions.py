import torch
import torch.nn.functional as F
from mazehelper.mazes_info import POSSIBLE_LOCATION_ACTION_FOR_MAZE_ID
from datahelper.load_data import load_optimal_behaviour


def transition_matrix_from_action_matrix(action_matrix, ny=7, nx=7):
    """
    Create a transition matrix from an action matrix.
    The action matrix should have the last dimension of size 4,
    representing the probabilities of moving right, up, left and down, respectively.

    The transition matrix will have shape (batch_size, ns, ns),
    where ns is the number of states (nx * ny).
    the indexing is done as follows: index i corresponds to state (x, y) where x = i // ny and y = i % ny.
    moving right increases the index by ny, moving up increases it by 1,
    moving left decreases it by ny, and moving down decreases it by 1.
    :param action_matrix: the action matrix with shape (batch_size, ns, 4)
    :type action_matrix: torch.Tensor
    :param ny: the number of rows in the grid (height)
    :type ny: int
    :param nx: the number of columns in the grid (width)
    :type nx: int
    :return: the transition matrix with shape (batch_size, ns, ns)
    :rtype: torch.Tensor
    """
    assert action_matrix.shape[-1] == 4, "Action matrix must have last dimension of size 4 (right, up, left, down)"
    if len(action_matrix.shape) == 2:
        action_matrix = action_matrix.unsqueeze(0)
    ns = nx * ny
    T = torch.zeros(action_matrix.shape[0], ns, ns)
    for i in range(ns):
        x, y = i // ny, i % ny
        if x + 1 < nx:
            T[:, i, i + ny] = action_matrix[:, i, 0]
        if y + 1 < ny:
            T[:, i, i + 1] = action_matrix[:, i, 1]
        if x - 1 >= 0:
            T[:, i, i - ny] = action_matrix[:, i, 2]
        if y - 1 >= 0:
            T[:, i, i - 1] = action_matrix[:, i, 3]
    if T.shape[0] == 1:
        T = T.squeeze(0)
    return T


def location_action_base_adjacency_matrix(nx=7, ny=7, reverse=True):
    """

    :param nx: the side length of the environment. 7 for the maze
    :type nx: int
    :param ny: the side length of the environment. 7 for the maze
    :type ny: int
    :param reverse:
    :return: finds location-action transition matrix where the index i
    corresponds to location action (s, a) where a = i // ns and s = i % ns.
    (x, y) = (s // ny, s % ny). n_s is the number of states (nx * ny) and
    n_sa is the number of state-action pairs (n_s * 4). Four actions (0, 1, 2, 3)
    correspond to (right, up, left, down). The transition matrix
    T is of shape (n_sa, n_sa). A location-action pair (s, a) can transition to
    another location-action pair (s1, a1) if the s1 is the state that can be reached
    as a result of taking action a from state s, and a1 is any action that can be taken
    from that location. Not if reverse is False, then the action a1 is not allowed to
    be the opposite of a.
    :rtype:
    """
    ns = nx * ny
    nsa = ns * 4
    T = []
    for i in range(nsa):
        s, a = i % ns, i // ns
        x, y = s // ny, s % ny
        if a == 0 and x + 1 < nx:
            s1 = s + ny
        elif a == 1 and y + 1 < ny:
            s1 = s + 1
        elif a == 2 and x - 1 >= 0:
            s1 = s - ny
        elif a == 3 and y - 1 >= 0:
            s1 = s - 1
        else:
            s1 = None
        if s1 is not None:
            if not reverse:
                possible_next_actions = torch.tensor([i for i in range(4) if i != (a + 2) % 4])
                possible_next_states = possible_next_actions * ns + s1
            else:
                possible_next_states = torch.arange(4) * ns + s1
            T.append(F.one_hot(possible_next_states, nsa).sum(dim=0))
        else:
            T.append(torch.zeros(nsa))
    T = torch.stack(T)
    return T


def location_action_adjacency_matrix_from_maze_id(maze_id, reverse=True):
    """
    Returns the adjacency matrix for the maze specified by maze_id.
    :param maze_id: the id of the maze, e.g. 1 for the first maze
    :type maze_id: int
    :param reverse: if True, the action a1 is allowed to be the opposite of a,
    :type reverse: bool
    :return: the adjacency matrix of shape (n_sa, n_sa) where n_sa is the number of
    state-action pairs (n_s * 4), where n_s is the number of states
    (nx * ny). The index i corresponds to location action (s, a) where
    a = i // ns and s = i % ns. (x, y) = (s // ny, s % ny).
    The four actions (0, 1, 2, 3) correspond to (right, up, left, down).
    The transition matrix T is of shape (n_sa, n_sa).
    A location-action pair (s, a) can transition to another location-action pair (s1, a1),
    if s1 is the state that can be reached as a result of taking action a from state s,
    and a1 is any action that can be taken from that location.
    Not if reverse is False, then the action a1 is not allowed to be the opposite of a.
    :rtype: torch.Tensor
    """
    T_mask = torch.tensor(POSSIBLE_LOCATION_ACTION_FOR_MAZE_ID[maze_id])
    maze_mask = T_mask.unsqueeze(-1) @ T_mask.unsqueeze(-2)  # what are the possible SAs in the maze
    adjacency_mask = location_action_base_adjacency_matrix(nx=7, ny=7, reverse=reverse)
    maze_SA_adjacency_matrix = maze_mask * adjacency_mask
    return maze_SA_adjacency_matrix


def location_action_straightish_transition_matrix(maze_id, straight_coef=3, remove_dead_ends=True, reversal_coef=1):
    """
    Generate a state-action transition matrix for a maze with a preference for straight actions.
    :param maze_id: identifier for the maze, e.g. 1 for the first maze
    :type maze_id: int
    :param straight_coef: coefficient to increase the probability of straight actions, 3 means that straight actions are (3 + 1) times more likely than turning left or right
    :type straight_coef: float
    :param remove_dead_ends: remove dead ends from the adjacency matrix, if True, the dead ends are removed,
    note this is done recursively so that no dead ends are left in the adjacency matrix
    :type remove_dead_ends: bool
    :param reversal_coef: coef to reduce the probability of reversal actions to zero, 1 means that reversal actions are not allowed, 0 means that reversal actions are fully allowed
    :type reversal_coef: float
    :return: a state-action transition matrix of shape (n_sa, n_sa) where n_sa is the number of state-action pairs (n_s * 4),
    where n_s is the number of states (nx * ny). The index i corresponds to location action (s, a) where
    a = i // ns and s = i % ns. (x, y) = (s // ny, s % ny).
    The four actions (0, 1, 2, 3) correspond to (right, up, left, down).
    The transition matrix T is of shape (n_sa, n_sa).
    :rtype: torch.Tensor
    """
    # get the state action transition matrix without dead ends
    # ------------------------------------------------------------------------------------------------------------------
    adjacency_mask = location_action_base_adjacency_matrix(nx=7, ny=7, reverse=True)
    adj = location_action_adjacency_matrix_from_maze_id(maze_id)
    if remove_dead_ends:
        dead_ends = (adj.sum(dim=-1) == 1).sum()
        while dead_ends > 0:
            sa_possible = (adj.sum(dim=-1) > 1).int()
            adj = (sa_possible.unsqueeze(-1) @ sa_possible.unsqueeze(-2)) * adjacency_mask
            dead_ends = (adj.sum(dim=-1) == 1).sum()

    # Increase probability for when the action is the same and reduce probability of reversal to zero
    # ------------------------------------------------------------------------------------------------------------------
    straight_action_addition = torch.kron(torch.eye(4), torch.ones(49, 49) * straight_coef)
    reverse_removal = torch.kron(F.one_hot((torch.arange(4) + 2) % 4), torch.ones(49, 49))
    straight_encourage = 1 + straight_action_addition - reverse_removal * reversal_coef
    G = adj * straight_encourage
    norm = G.sum(dim=-1, keepdims=True)
    norm[norm == 0] = 1  # if the norm is zero, get rid of it
    G = G / norm
    return G


def location_action_vector_transition_matrix(maze_id, inverse_temp=1):
    """
    Generate a location-action transition matrix for a maze with a preference for actions that move towards the goal using vector-based navigation.
    The preference is controlled by the inverse temperature parameter, with higher values leading to a stronger preference.
    :param maze_id: identifier for the maze, e.g. 1 for the first maze
    :type maze_id: int
    :param inverse_temp: inverse temperature parameter, higher values lead to a stronger preference for actions that move towards the goal
    :type inverse_temp: float
    :return: a state-action transition matrix of shape (n_sa, n_sa) where n_sa is the number of state-action pairs (n_s * 4),
    where n_s is the number of states (nx * ny). The index i corresponds to location action (s, a) where
    a = i // ns and s = i % ns. (x, y) = (s // ny, s % ny).
    The four actions (0, 1, 2, 3) correspond to (right, up, left, down).
    :rtype: torch.Tensor
    """
    # get the state action transition matrix without dead ends
    # ------------------------------------------------------------------------------------------------------------------
    adj = location_action_adjacency_matrix_from_maze_id(maze_id)

    x = torch.arange(49)[:, None] // 7 - torch.arange(49)[None, :] // 7  # difference in x between all possible positions
    y = torch.arange(49)[:, None] % 7 - torch.arange(49)[None, :] % 7  # difference in y between all possible positions

    cos_theta = x / torch.sqrt(x ** 2 + y ** 2 + 1e-12)  # cosine of the angle between the two points 49 x 49
    sin_theta = y / torch.sqrt(x ** 2 + y ** 2 + 1e-12)  # sine of the angle between the two points 49 x 49

    spatial_mask = torch.stack([cos_theta, sin_theta,-cos_theta, -sin_theta], dim=-1)  # 49 x 49 x 4
    spatial_mask = F.softmax(spatial_mask * inverse_temp, dim=-1)  # softmax the spatial mask, so all actions sum to 1 for each location pair
    # softmax is applied here so that there's no need to deal with zeros in the adjacency matrix.
    spatial_mask = spatial_mask.transpose(-1, -2).flatten(-2, -1)[:, None, :].repeat(1, 196, 1) # .repeat(1, 196).reshape(-1, 196, 196)
    # translate this to reward x location_action x location_action matrix, 49 x 196 x 196, first dimension is reward

    G = adj * spatial_mask
    G = F.normalize(G, dim=-1)
    return G


def location_action_optimal_transition_matrix(maze_id):
    """
    Generate a location-action transition matrix for a maze using the optimal policy.
    :param maze_id: identifier for the maze, e.g. 1 for the first maze
    :type maze_id: int
    :return: a state-action transition matrix of shape (n_s, n_sa, n_sa) where n_sa is the number of state-action pairs (n_s * 4),
    where n_s is the number of states (nx * ny). The index i corresponds to location action (s, a) where
    a = i // ns and s = i % ns. (x, y) = (s // ny, s % ny).
    The four actions (0, 1, 2, 3) correspond to (right, up, left, down).
    """
    optimal_policy = torch.tensor(load_optimal_behaviour(maze_id))
    adj = location_action_adjacency_matrix_from_maze_id(maze_id)
    optimal_policy_sa = optimal_policy.transpose(-2, -1).flatten(-2, -1)[:, None, :].repeat(1, 196, 1)
    optimal_policy_sa = optimal_policy_sa * adj.unsqueeze(0)
    return optimal_policy_sa


def location_2_action_optimal_transition_matrix(maze_id):
    """
    Generate a location to action transition matrix for a maze using the optimal policy.
    :param maze_id: identifier for the maze, e.g. 1 for the first maze
    :type maze_id: int
    :return: a location to action transition matrix of shape (n_s, n_a) where n_s is the number of states (nx * ny).
    Index i for state s corresponds to (x, y) where x = i // ny and y = i % ny.
    The four actions (0, 1, 2, 3) correspond to (right, up, left, down).
    :rtype: torch.Tensor
    """
    optimal_policy = torch.tensor(load_optimal_behaviour(maze_id))
    return optimal_policy


def location_2_action_vector_transition_matrix(maze_id, inverse_temp=1):
    """
    Generate a location to action transition matrix for a maze with a preference for actions that move towards the goal using vector-based navigation.
    The preference is controlled by the inverse temperature parameter, with higher values leading to a stronger preference.
    :param maze_id: identifier for the maze, e.g. 1 for the first maze
    :type maze_id: int
    :param inverse_temp: inverse temperature parameter, higher values lead to a stronger preference for actions that move towards the goal
    :type inverse_temp: float
    :return: a location to action transition matrix of shape (n_s, n_a)
    where n_s is the number of states (nx * ny). Index i for state s corresponds to (x, y) where x = i // ny and y = i % ny.
    The four actions (0, 1, 2, 3) correspond to (right, up, left, down).
    :rtype: torch.Tensor
    """
    # get the state action transition matrix without dead ends
    # ------------------------------------------------------------------------------------------------------------------
    spatial_mask = torch.tensor(POSSIBLE_LOCATION_ACTION_FOR_MAZE_ID[maze_id]).reshape(4, 49).t()  # what are the possible actions in the maze

    x = torch.arange(49)[:, None] // 7 - torch.arange(49)[None, :] // 7  # difference in x between all possible positions
    y = torch.arange(49)[:, None] % 7 - torch.arange(49)[None, :] % 7  # difference in y between all possible positions

    cos_theta = x / torch.sqrt(x ** 2 + y ** 2 + 1e-12)  # cosine of the angle between the two points 49 x 49
    sin_theta = y / torch.sqrt(x ** 2 + y ** 2 + 1e-12)  # sine of the angle between the two points 49 x 49

    policy = torch.stack([cos_theta, sin_theta,-cos_theta, -sin_theta], dim=-1)  # 49 x 49 x 4
    policy = F.softmax(policy * inverse_temp, dim=-1)  # softmax the spatial mask, so all actions sum to 1

    policy = policy * spatial_mask
    policy = F.normalize(policy, p=1, dim=-1)
    return policy