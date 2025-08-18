import torch
from torch.distributions import Gamma, Categorical
from mazehelper.transition_matrix_functions import *


def random_route_generate(maze_id, n_routes=1, straight_coef=3, route_length_mean=None, route_length_var=None, reduce_overlap=False):
    """
    Generates random routes in the maze with a given maze_id.
    :param maze_id: int, identifier of the maze (1, 2, or 3)
    :param n_routes: int, number of routes to generate
    :param straight_coef: int, coefficient for the straightness of the route
    :param route_length_mean: float, mean of the route length distribution, if None, a default value is used
    :param route_length_var: float, variance of the route length distribution, if None, a default value is used
    :param reduce_overlap: bool, if True, reduces overlap between routes
    :type reduce_overlap: bool
    :return: tuple of (route_flat, route_length)
    - route_flat: tensor of shape (n_routes, 196) where each row represents a route in the maze
    - route_length: tensor of shape (n_routes,) representing the length of each route
    """
    if reduce_overlap:
        n_routes = n_routes * 2  # if reducing overlap, generate twice as many routes

    if route_length_mean is None or route_length_var is None:
        route_length_dist = Gamma(71, 5, n_routes)  # mean = (71)/ 5, variance = 71/5**2. Gamma ensures that route dist
    else:
        route_length_dist = Gamma(route_length_mean ** 2 / route_length_var, route_length_mean / route_length_var, n_routes)  # mean = a / b, variance =a / b**2, a and b are shape and scale parameters
    route_length = route_length_dist.sample((n_routes,)).round()
    # is never less than zero

    # create mask with route length given by route_length_dist. Allow for parallel sampling
    # -----------------------------------------------------------------------------------------------------------------
    max_route_length = route_length.max().int()
    T = torch.arange(max_route_length).repeat(n_routes).reshape(n_routes, max_route_length)
    G = location_action_straightish_transition_matrix(maze_id, straight_coef=straight_coef, reversal_coef=1, remove_dead_ends=True)
    sa_possible = G.sum(dim=-1)

    # adjacency matrix of the maze, where actions going straight have a higher probability. Some routes will terminate
    # early due to going into a dead end
    # ------------------------------------------------------------------------------------------------------------------
    sa = Categorical(sa_possible).sample((n_routes,))
    route = [sa] # initialise the route with the first action sampled from the possible actions
    for i in range(max_route_length-1): # for each step in the route
        sa = Categorical(G[sa]).sample() # sample the next action based on the current action and the adjacency matrix
        route.append(sa) # append the sampled action to the route
    route = torch.stack(route).transpose(-2, -1) # shape (n_routes, max_route_length)

    # find routes with the least overlap if reduce_overlap is True
    # ------------------------------------------------------------------------------------------------------------------
    if reduce_overlap:
        routes_one_hot = F.one_hot(route, 196).sum(dim=-2)
        overlap = routes_one_hot @ routes_one_hot.t()  # no need to normalize as all routes have the same length and are one-hot encoded

        # chose a random route to start with, then iteratively choose the next route that has the least overlap with the already chosen routes
        # ---------------------------------------------------------------------------------------------------------------------------------------------------
        routes_with_small_overlap = [torch.randint(n_routes, (1,))]  # start with a random route
        for _ in range(n_routes // 2 - 1):
            _, next_route = overlap[routes_with_small_overlap, :].max(dim=0)[0].min(dim=-1)
            routes_with_small_overlap.append(next_route)
        routes_with_small_overlap = torch.tensor(routes_with_small_overlap)
    mask = T < route_length.unsqueeze(-1)
    route_flat = ((F.one_hot(route, 196)) * mask.unsqueeze(-1)).sum(dim=-2)
    if reduce_overlap:
        route_flat = route_flat[routes_with_small_overlap]
        route_length = route_length[routes_with_small_overlap]
    return route_flat, route_length

