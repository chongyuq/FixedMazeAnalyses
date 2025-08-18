import json
import os

import numpy as np
import torch
from torch.distributions import Categorical
import pandas as pd

from lowrank_lmdp.model import LowRankLMDP
from mazehelper.transition_matrix_functions import *
from mazehelper.mazes_info import POSSIBLE_LOCATION_ACTION_FOR_MAZE_ID

from synthetic_data_generate.route_generation_funcs import random_route_generate


def generate_synthetic_agents(root_dir, agent_type, maze_number, agent_number):
    """
    Generate synthetic agents based on the specified type and maze number.

    :param root_dir: Root directory for data storage.
    :param agent_type: Type of agent to generate (e.g., 'non_markovian_agents', 'markovian_agents', etc.).
    :param maze_number: The maze number for which to generate the agent.
    :param agent_number: The specific agent number to generate.
    """
    steps_data = pd.read_csv(f'{root_dir}/data/time_at_node.csv')
    time_data = pd.read_csv(f'{root_dir}/data/trial_session_ITI_steps.csv')

    generate_agent(root_dir, agent_type, maze_number, agent_number)
    return


def generate_lmdp_agent(maze_number, agent_number,
                        steps_distribution, ITI_steps_distribution, trial_time_at_node_distribution,
                        ITI_time_at_node_distribution, reward_time_at_node_distribution,):
    """
    Generate an LMDP agent for a specific maze and agent number.

    :param root_dir: Root directory for data storage.
    :param maze_number: The maze number for which to generate the agent.
    :param agent_number: The specific agent number to generate.
    """
    # agent metadata
    # -------------------------------------------------------------------------------
    n_routes = np.random.randint(3, 6)  # randomly choose number of routes
    straight_coef = np.random.rand() * 4  # randomly choose a straight coefficient
    route_length_mean = np.random.rand() * 5 + 9  # mean route length
    route_length_var = np.random.randint(2, 6)
    cognitive_constant = np.random.rand() * 20 + 10  # cognitive constant
    action_cost = np.random.rand() * 0.05 + 0.1
    reward_value = np.random.rand() * 2 + 1  # reward value

    # generate routes
    # ------------------------------------------------------------------------------- # variance of route
    routes, route_length = random_route_generate(maze_id=maze_number,
                                                 n_routes=n_routes,
                                                 straight_coef=straight_coef,
                                                 route_length_mean=route_length_mean,
                                                 route_length_var=route_length_var,
                                                 reduce_overlap=True)

    # create the LMDP agent
    # -------------------------------------------------------------------------------
    adjacency_matrix = location_action_adjacency_matrix_from_maze_id(maze_id=maze_number)
    planner = LowRankLMDP(n_locs=49,
                          n_acts_per_loc=4,
                          cognitive_constant=cognitive_constant,
                          action_cost=action_cost,
                          reward_value=reward_value,
                          adjacency_matrix=adjacency_matrix,
                          routes=routes)


    # define relevant variables to match the behaviour of mice
    # -----------------------------------------------------------------------------------------------------------------
    simulated_steps = steps_distribution.sample()

    # simulate the agent's behaviour
    # -----------------------------------------------------------------------------------------------------------------
    trajectories, rewards, times, trials, trial_phases, starts = [], [], torch.tensor([]), [], [], []
    locs = Categorical(torch.ones(49))  # locations are uniformly distributed
    start = locs.sample()
    steps, trial = 0, 0
    prev_reward = None

    while steps < simulated_steps:
        finish = locs.sample()
        while finish == start or finish == prev_reward:
            finish = locs.sample()
        trajectory = planner.generate_trajectory(start, finish)
        n_steps = len(trajectory)
        steps += n_steps

        # sample time for each trial step and for the reward time
        # ------------------------------------------------------------------------------------------------------
        trial_time = trial_time_at_node_distribution.sample((n_steps,))
        reward_time = reward_time_at_node_distribution.sample()
        trial_time[-1] = reward_time
        trial_phase = ['navigation'] * n_steps

        # sample the number of ITI steps. Diffusion with no reverse and preference for forward direction
        # ------------------------------------------------------------------------------------------------------
        n_ITI_steps = ITI_steps_distribution.sample().round().int()
        steps += n_ITI_steps
        s_prev, a_prev = finish, trajectory[-1] % 196 // 49
        s_new = s_prev + (((a_prev % 2) == 0) * 7 + ((a_prev % 2) == 1) * 1) * \
                ((a_prev <= 1) * 1 + (a_prev > 1) * -1)
        iti_steps, iti_times = iti_steps_generate(maze_id=maze_number,
                                                  start=s_new,
                                                  n_steps=n_ITI_steps,
                                                  iti_steps_dist=ITI_time_at_node_distribution,
                                                  straight_coef=straight_coef,
                                                  reverse_coef=0.999)
        if n_ITI_steps > 0:
            trial_phase.extend(['ITI'] * n_ITI_steps)
            trial_time = torch.cat([trial_time, iti_times])
            trajectory.extend([-196 + i for i in iti_steps])  # allocate route index = -1 n_routes + 2 to ITI

        # store the trajectory data
        # ------------------------------------------------------------------------------------------------------
        trial += 1
        trajectories.extend(trajectory)
        rewards.extend([finish] * len(trajectory))
        starts.extend([start] * len(trajectory))
        times = torch.cat([times, trial_time])
        trials.extend([trial] * len(trajectory))
        trial_phases.extend(trial_phase)
        start = s_new  # update new location
        prev_reward = finish

    # data to save
    # -------------------------------------------------------------------------------
    agent_data = {
        'maze_id': maze_number,
        'unique_id': f'{maze_number}_{agent_number}',
        'n_routes': n_routes,
        'straight_coef': straight_coef,
        'route_length_mean': route_length_mean,
        'route_length_var': route_length_var,
        'agent_number': agent_number,
        'agent_type': 'lmdp_agents',
        'cognitive_constant': cognitive_constant,
        'action_cost': action_cost,
        'reward_value': reward_value,
        'routes': routes.tolist(),
        'route_length': route_length.tolist()
    }

    trajectories = torch.tensor(trajectories)
    rewards = torch.tensor(rewards)
    starts = torch.tensor(starts)
    times = torch.cumsum(times, dim=-1)
    trials = torch.tensor(trials)
    pos_idx = trajectories % 196 % 49
    action_class = trajectories % 196 // 49
    route = trajectories // 196

    agent_trajectories_data = pd.DataFrame({
        'maze_number':maze_number,
        'unique_id': f'{maze_number}_{agent_number}',
        'subject_ID': agent_number,
        'trial': trials,
        'time': times,
        'trial_phase': trial_phases,
        'reward_idx': rewards,
        'pos_idx': pos_idx,
        'start_idx': starts,
        'action_class': action_class,
        'route': route,
    })
    return agent_data, agent_trajectories_data


def iti_steps_generate(maze_id, start, n_steps, iti_steps_dist, straight_coef=3, reverse_coef=0.999):
    """
    Generate ITI steps for the agent's behaviour.

    :param maze_id: ID of the maze for which to generate ITI steps.
    :param start: Starting location for the ITI steps
    :param n_steps: Number of ITI steps to generate.
    :param iti_steps_dist: Distribution for ITI steps.
    :param straight_coef: Coefficient for straight movement. 3 means (3+1) times more likely to turn than to go straight.
    :param reverse_coef: Coefficient for reverse movement. 1 means no reverse movement, 0 means reverse movement is as likely as turning.
    :return: List of ITI steps where each step is an index for a location-action pair, starting from the given location
    the index is calculated as (action * 49 + location), where action is the action taken and location is the current location.
    If n_steps is 0, returns an empty list.
    generates a list of ITI steps based on the specified distribution and coefficients.
    , list of times for each step
    """
    diffusion = 1 + torch.eye(4) * straight_coef - torch.eye(4).roll(2, dims=0) * reverse_coef
    # diffusion encourages forward direction, almost removes backward
    action_matrix = POSSIBLE_LOCATION_ACTION_FOR_MAZE_ID[maze_id].reshape(4, 49).t()
    iti_steps = []
    s_new = start
    if n_steps > 0:
        ITI_time = iti_steps_dist.sample((n_steps,))
        for _ in range(n_steps):
            a_new = Categorical(action_matrix[s_new] * diffusion[a_prev]).sample()
            iti_steps.append(a_new * 49 + s_new)
            s_prev, a_prev = s_new, a_new
            s_new = s_prev + (((a_prev % 2) == 0) * 7 + ((a_prev % 2) == 1) * 1) * \
                    ((a_prev <= 1) * 1 + (a_prev > 1) * -1)
    return iti_steps, ITI_time

