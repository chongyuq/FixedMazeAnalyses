import torch
import torch.nn as nn
from torch.distributions import Categorical
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from mazehelper.mazes_info import POSSIBLE_LOCATION_ACTION_FOR_MAZE_ID
from mazehelper.transition_matrix_functions import location_action_vector_transition_matrix, \
    location_action_optimal_transition_matrix, location_action_straightish_transition_matrix


class DijkstraRoutePlanner:
    def __init__(self,
                 n_locs: int,
                 n_acts_per_loc: int,
                 cognitive_constant: float,
                 adjacency_matrix: torch.Tensor,
                 routes: torch.Tensor,
                 inverse_temperature: float = 1.0
                 ):
        """
        DijkstraRoutePlanner uses Dijkstra's algorithm to find the shortest path in a graph with given routes and cognitive costs.
        :param n_locs: number of locations in the maze
        :type n_locs: int
        :param n_acts_per_loc: number of actions per location
        :type n_acts_per_loc: int
        :param cognitive_constant: how much more costly it is to switch routes compared to staying on the same route
        :type cognitive_constant: float
        :param adjacency_matrix: adjacency matrix of the graph, shape (n_states, n_states)
        :type adjacency_matrix: torch.Tensor
        :param routes: routes in the maze, shape (n_routes, n_states), where n_states = n_locs * n_acts_per_loc, 0 for location action not on route
        and 1 for location action on route
        :type routes: torch.Tensor
        :param inverse_temperature: inverse temperature for the softmax distribution over the starting and ending state-route conjunctions
        :type inverse_temperature: float
        """
        self.n_locs = n_locs
        self.n_acts_per_loc = n_acts_per_loc
        self.n_states = n_locs * n_acts_per_loc  # total number of states in the graph
        self.n_routes = routes.shape[0]  # number of routes
        self.P_size = (self.n_routes + 1) * self.n_states  # size of the transition matrix
        self.cognitive_constant = torch.tensor(cognitive_constant)
        self.inverse_temperature = inverse_temperature

        self.adjacency_matrix = adjacency_matrix  # adjacency matrix of the graph. Row index is the source state, column index is the destination state.
        self.mask = (self.adjacency_matrix.sum(
            dim=-1) > 0).float()  # mask for the states that have at least one outgoing edge
        assert adjacency_matrix.shape == (self.n_states, self.n_states), \
            "Adjacency matrix must be of shape (n_states, n_states)"

        # calculate the route matrix where the numbers represent distance between location action routes
        # ---------------------------------------------------------------------------------------------------------------------
        self.routes = torch.cat([routes, self.mask.unsqueeze(0)])
        self.route_matrix = torch.einsum('rs, RS -> rRsS', self.routes, self.routes)  # shape n_route x n_route x sa x sa
        # self.route_matrix = self.routes.unsqueeze(0).unsqueeze(-2).repeat(self.n_routes + 1, 1, self.n_states,
        #                                                                             1) # shape (n_routes + 1, n_routes + 1, n_states, n_states)
        # all off diagonals are equal to 1 + cognitive constant, diagonals are equal to 1
        # ---------------------------------------------------------------------------------------------------------------------
        self.cognitive_cost_matrix = torch.ones(self.n_routes + 1, self.n_routes + 1) - (torch.eye(self.n_routes + 1) - 1) * self.cognitive_constant
        self.cognitive_cost_matrix[-1, -1] = 1 + self.cognitive_constant

        # calculate the transition matrix with distances
        # ---------------------------------------------------------------------------------------------------------------------
        self.P = self.route_matrix * self.adjacency_matrix * self.cognitive_cost_matrix[..., None, None]
        self.P = self.P.transpose(1, 2).flatten(0, 1).flatten(-2, -1)

        # use Dijkstra's algorithm to calculate the shortest path from each state to each state
        # ---------------------------------------------------------------------------------------------------------------------
        cgraph = csr_matrix(self.P.numpy())
        route_distance_matrix, predecessors = shortest_path(cgraph, directed=True, return_predecessors=True)  # within route shortest distance
        self.route_distance_matrix, self.predecessors = torch.tensor(route_distance_matrix), torch.tensor(predecessors)


    def get_all_location_action_route_conj(self, location):
        """Get all location-action-route conjunctions for a given location. Location is the index of the location"""
        states = torch.arange(self.n_acts_per_loc) * self.n_locs + location
        state_routes = torch.arange(self.n_routes + 1)[:, None].repeat(1, self.n_acts_per_loc) * self.n_states + states
        state_routes = state_routes.flatten()
        return state_routes

    def starting_and_ending_state_route(self, reward, location):
        """
        Starting state route returns the starting state and route for a given reward and location. If location is None,
        It will sample from the whole distribution include the distribution across locations
        :param reward: reward index, shape (1,), defines the reward state, dtype torch.int64
        :type reward: torch.Tensor
        :param location: location index, shape (1,), defines the location, dtype torch.int64
        :type location: torch.Tensor, optional
        :return:
        :rtype:
        """
        # get all possible starting state-route conjunctions for the given location
        # -----------------------------------------------------------------------------------------------------
        starting_state_routes = self.get_all_location_action_route_conj(location)
        ending_state_routes = self.get_all_location_action_route_conj(reward)

        # sample from the distribution over the starting state-route conjunctions to get the starting state and route
        # -----------------------------------------------------------------------------------------------------
        start_end_state_route_index = Categorical(torch.exp(-self.inverse_temperature * self.route_distance_matrix[starting_state_routes][:, ending_state_routes].flatten())).sample()
        starting_state_route, ending_state_route = starting_state_routes[start_end_state_route_index // ending_state_routes.size(0)], ending_state_routes[
            start_end_state_route_index % ending_state_routes.size(0)]

        return starting_state_route, ending_state_route


    def generate_trajectory(self, start_location, reward, max_length=300):
        """Generate a trajectory from a starting location to a reward state.
        :param start_location: starting location index, between 0 and n_locs - 1
        :type start_location: int
        :param reward: reward location index, between 0 and n_locs - 1
        :type reward: int
        :return: trajectory as a list of tuples (state, route)
        where state is the index of the state (location-action pair) and route is the index of the route
        index of location-action pair is defined as state = action * n_locs + location
        index of x and y coordinates in the maze is given by:
        (x, y) = (location// 7, location % 7)
        :rtype: list of tuples
        """
        starting_state_route, ending_state_route = self.starting_and_ending_state_route(reward, start_location)
        trajectory = [((ending_state_route % self.n_states).item(), (ending_state_route // self.n_states).item())]  # list of tuples(state, route)
        terminal = False
        prev_step = ending_state_route
        while terminal is False and len(trajectory) < max_length:
            prev_step = self.predecessors[starting_state_route, prev_step]
            trajectory.insert(0, ((prev_step % self.n_states).item(), (prev_step // self.n_states).item()))
            if prev_step == starting_state_route:
                terminal = True
        return trajectory


class VectorOptimalForwardPlanner(nn.Module):
    def __init__(self, maze_number: int, vector_coef: float, optimal_coef: float, straight_coef: float):
        super().__init__()
        self.maze_number = maze_number
        self.vector_coef = vector_coef
        self.optimal_coef = optimal_coef
        self.straight_coef = straight_coef
        self.vector_matrix = location_action_vector_transition_matrix(maze_id=maze_number)
        self.optimal_matrix = location_action_optimal_transition_matrix(maze_id=maze_number)
        self.straight_matrix = location_action_straightish_transition_matrix(maze_id=maze_number, straight_coef=4, reversal_coef=0.999, remove_dead_ends=False)
        # note that the straight_matrix here is only used to encourage going straight, straight_coef is not related to the straight_coef used in route generation
        # it is set to a high value to ensure that the agent goes straight as much as possible, 4 means that going straight is 4 + 1 times more likely than turning
        self.T = (self.vector_matrix * vector_coef + self.optimal_matrix * optimal_coef + self.straight_matrix * straight_coef) / (vector_coef + optimal_coef + straight_coef)
        self.T_no_straight = (self.vector_matrix * vector_coef + self.optimal_matrix * optimal_coef) / (vector_coef + optimal_coef)
        # shape of self.T is 49 x 196 x 196, where T[i, j, k] is the probability of going to location-action k from location-action j when the reward is at location i
        return

    def forward(self, state, reward):
        sa = Categorical(self.T[reward, state]).sample()
        terminal = (sa % 49) == reward
        return sa, terminal

    def starting_state(self, reward, location, prev_action=None):
        """generates a starting location-action pair given a reward location and a starting location
        as it is the first action (when prev_action is None), there is no contribution from 'going straight'. This is done by chosing any
        available location-action that leads to the starting location and sampling a location-action pair from there.
        However, if prev_action is given, then the 'going straight' contribution is included.
        """
        if prev_action is None:
            possible_previous_location_action = [
                0 * 49 + (location - 7),  # right
                1 * 49 + (location - 1),  # up
                2 * 49 + (location + 7),  # left
                3 * 49 + (location + 1)   # down
            ]
            possible_previous_location_action = [x for x in possible_previous_location_action if 0 <= x < 196 and POSSIBLE_LOCATION_ACTION_FOR_MAZE_ID[self.maze_number][x] == 1]
            sa = Categorical(self.T_no_straight[reward, possible_previous_location_action[0]]).sample()
            return sa
        else:
            if prev_action == 0:
                prev_location = location - 7
            elif prev_action == 1:
                prev_location = location - 1
            elif prev_action == 2:
                prev_location = location + 7
            elif prev_action == 3:
                prev_location = location + 1
            previous_location_action = prev_action * 49 + prev_location
            assert 0 <= previous_location_action < 196 and POSSIBLE_LOCATION_ACTION_FOR_MAZE_ID[self.maze_number][previous_location_action] == 1, "Invalid previous location-action pair"
            sa = Categorical(self.T[reward, previous_location_action]).sample()
            return sa


    def generate_trajectory(self, starting_location, reward, previous_action=None, max_length=300):
        state = self.starting_state(reward, starting_location, previous_action)
        trajectory = [state.item()]
        for _ in range(max_length):
            next_state, terminal = self.forward(state, reward)
            trajectory.append(next_state.item())
            if terminal:
                break
            state = next_state
        return trajectory







