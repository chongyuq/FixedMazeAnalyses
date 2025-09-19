import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from mazehelper.mazes_info import POSSIBLE_LOCATION_ACTION_FOR_MAZE_ID
from mazehelper.transition_matrix_functions import location_action_vector_transition_matrix, \
    location_action_optimal_transition_matrix, location_action_straightish_transition_matrix, \
    location_2_action_vector_transition_matrix, location_2_action_optimal_transition_matrix


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


class VectorOptimalExponentialActionPlanner(nn.Module):
    def __init__(self, maze_number: int, vector_coef: float, optimal_coef: float, straight_coef: float, inverse_temp: float = 1.0, alpha: float = 0.2):
        super().__init__()
        self.maze_number = maze_number
        self.vector_coef = vector_coef / (vector_coef + optimal_coef + straight_coef)
        self.optimal_coef = optimal_coef / (vector_coef + optimal_coef + straight_coef)
        self.straight_coef = straight_coef / (vector_coef + optimal_coef + straight_coef)

        self.inverse_temp = inverse_temp
        self.alpha = torch.tensor(alpha)

        self.vector_matrix = location_2_action_vector_transition_matrix(maze_id=maze_number, inverse_temp=inverse_temp)
        self.optimal_matrix = location_2_action_optimal_transition_matrix(maze_id=maze_number)
        self.T_no_straight = self.vector_matrix * vector_coef + self.optimal_matrix * optimal_coef
        # shape of self.T is 49 x 49 x 4 where T[i, j, k] is the probability of taking action k from location j when the reward is at location i

        self.reverse_matrix = torch.eye(4).roll(2, dims=0)
        self.possible_actions = torch.tensor(POSSIBLE_LOCATION_ACTION_FOR_MAZE_ID[self.maze_number]).reshape(4, 49).t()
        # shape 4 x 4, where reverse_matrix[i, j] = 1 if action j is the reverse of action i, else 0
        return

    def forward(self, state, reward, action_exp):
        """ note that state here refers to location, not location-action pair"""
        action_exp_n = F.normalize(action_exp, p=1, dim=-1)
        action_exp_reverse_n = action_exp_n @ self.reverse_matrix
        action_mod = F.softmax((action_exp_n - action_exp_reverse_n) * self.inverse_temp, dim=-1)  # ensure that other actions are encouraged but less
        action_mod_possible = F.normalize(action_mod * self.possible_actions[state], p=1, dim=-1)

        if reward is None:
            a = Categorical(action_mod_possible.relu()).sample()
        else:
            a = Categorical((self.T_no_straight[reward, state] + self.straight_coef * action_mod_possible).relu()).sample()

        # update the state and the action exponential decay
        # --------------------------------------------------------------------------------------------------------------
        state_next = state + ((a % 2 == 0) * 7 + (a % 2 == 1) * 1) * ((a <= 1) * 1 + (a > 1) * -1)
        action_exp = self.action_exp_calculate(state, a, action_exp)
        terminal = state_next == reward
        return a, state_next, terminal, action_exp

    def generate_trajectory(self, start_location, reward, action_exp=None, max_length=100):
        """

        :param start_location: state start
        :type start_location: torch.Tensor.int
        :param reward: ending state
        :type reward: torch.Tensor.int
        :param T: trajectory matrix
        :type T: shape = 49 x 4
        :param alpha: decay
        :type alpha: torch.Tensor
        :param frac: how much to effect the transition matrix
        :type frac: torch.Tensor
        :param action_exp: exponential decay of previous actions where max needs to be 1.0
        :type action_exp: torch.Tensor  shape = 4
        :return: state action trajectory
        :rtype:
        """
        if action_exp is None:
            action_exp = torch.zeros(4)
        trajectory = []
        state = start_location
        for i in range(max_length):
            if state < 0 or state >= 49:
                raise ValueError(f"State {state} is out of bounds. It should be between 0 and 48.")
                print(state)
            action, next_s, terminal, action_exp = self.forward(state, reward, action_exp)
            trajectory.append((49 * action + state).item())
            state = next_s
            if terminal:
                action, _, _, action_exp = self.forward(state=state, reward=None, action_exp=action_exp)
                trajectory.append((49 * action + state).item())
                break
            if i == max_length - 1:
                action, _, _, action_exp = self.forward(state=state, reward=reward, action_exp=action_exp)
                trajectory.append((49 * action + state).item())
        return trajectory, action_exp

    def action_exp_calculate(self, state, action, action_exp):
        if self.possible_actions[state].sum() == 1:
            action_exp = F.one_hot(action, 4).float()  # reset the action exponential decay if a dead end
        else:
            action_exp = torch.exp(-self.alpha) * action_exp + F.one_hot(action, 4).float()
        return action_exp


