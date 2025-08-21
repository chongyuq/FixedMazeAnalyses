import json
import os

import numpy as np
import torch
from torch.distributions import Categorical, Gamma, LogNormal
import pandas as pd

from lowrank_lmdp.model import LowRankLMDP
from mazehelper.transition_matrix_functions import *
from mazehelper.mazes_info import POSSIBLE_LOCATION_ACTION_FOR_MAZE_ID

from synthetic_data_generate.route_generation_funcs import random_route_generate


def generate_synthetic_agents(root_dir, agent_type, maze_number, n_agents=6):
    """
    Generate synthetic agents based on the specified type and maze number.

    :param root_dir: Root directory for data storage.
    :param agent_type: Type of agent to generate (e.g., 'non_markovian_agents', 'markovian_agents', etc.).
    :param maze_number: The maze number for which to generate the agent.
    :param n_agents: Number of agents to generate.
    """
    time_data = pd.read_csv(f'{root_dir}/data/behaviour/summary_statistics/time_at_node.csv')
    steps_data = pd.read_csv(f'{root_dir}/data/behaviour/summary_statistics/trial_session_ITI_steps.csv')

    # Check if the maze_number exists in the data and check agent types
    # -----------------------------------------------------------------------------
    assert maze_number < 4 and maze_number > 0, "Maze number must be between 1 and 3."
    if agent_type not in ['lmdp_agents', 'dijkstra_agents', 'non_markovian_agents', 'markovian_agents']:
        raise ValueError(f"Unsupported agent type: {agent_type}. Supported types are: "
                         "'lmdp_agents', 'dijkstra_agents', 'non_markovian_agents', 'markovian_agents'.")

    # Calculate distributions for total steps for a maze
    # -----------------------------------------------------------------------------
    steps_mean, steps_std = steps_data[steps_data.maze_number == maze_number].total_steps.mean(), \
                            steps_data[steps_data.maze_number == maze_number].total_steps.std()
    beta_steps = steps_mean / steps_std ** 2
    alpha_steps = steps_mean * beta_steps
    steps_distribution = Gamma(alpha_steps, beta_steps)

    # Calculate distributions for ITI steps
    # -----------------------------------------------------------------------------
    ITI_steps_mean, ITI_steps_std = steps_data[steps_data.maze_number == maze_number].ITI_steps_mean.mean(), \
                                    steps_data[steps_data.maze_number == maze_number].ITI_steps_std.mean()  # note that the standard deviation is the mean of the standard deviation across subjects
    beta_steps_ITI = ITI_steps_mean / ITI_steps_std ** 2
    alpha_steps_ITI = ITI_steps_mean * beta_steps_ITI
    ITI_steps_distribution = Gamma(alpha_steps_ITI, beta_steps_ITI)

    # Calculate distributions for time at nodes, this is done separately for reward, navigation and ITI
    # -----------------------------------------------------------------------------
    reward_log_mean, navigation_log_mean, iti_log_mean = time_data[time_data.maze_number == maze_number].reward_log_mean.mean(), \
                                                    time_data[time_data.maze_number == maze_number].navigation_log_mean.mean(), \
                                                    time_data[time_data.maze_number == maze_number].iti_log_mean.mean()
    reward_log_std, navigation_log_std, iti_log_std = time_data[time_data.maze_number == maze_number].reward_log_std.mean(), \
                                                  time_data[time_data.maze_number == maze_number].navigation_log_std.mean(), \
                                                  time_data[time_data.maze_number == maze_number].iti_log_std.mean()
    reward_time_at_node_distribution = LogNormal(reward_log_mean, reward_log_std)
    navigation_time_at_node_distribution = LogNormal(navigation_log_mean, navigation_log_std)
    ITI_time_at_node_distribution = LogNormal(iti_log_mean, iti_log_std)


    if agent_type == 'lmdp_agents':
        lmdp_agents_meta_data, lmdp_agents_trajectories_dfs = [], []
        for agent_number in range(1, n_agents + 1):
            # Generate LMDP agent
            a, b = generate_lmdp_agent(maze_number=maze_number, agent_number=agent_number,
                                       steps_distribution=steps_distribution,
                                       ITI_steps_distribution=ITI_steps_distribution,
                                       navigation_time_at_node_distribution=navigation_time_at_node_distribution,
                                       ITI_time_at_node_distribution=ITI_time_at_node_distribution,
                                       reward_time_at_node_distribution=reward_time_at_node_distribution)
            lmdp_agents_meta_data.append(a)
            lmdp_agents_trajectories_dfs.append(b)
        lmdp_agents_meta_data = pd.DataFrame(lmdp_agents_meta_data)
        lmdp_agents_trajectories_df = pd.concat(lmdp_agents_trajectories_dfs)
        # save the LMDP agents data
        # -----------------------------------------------------------------------------
        lmdp_agents_meta_data.to_csv(f'{root_dir}/data/synthetic_data/lmdp_agents_meta_data_{maze_number}.csv', index=False)
        lmdp_agents_trajectories_df.to_csv(f'{root_dir}/data/synthetic_data/lmdp_agents_maze_{maze_number}.csv', index=False)
        print(f'LMDP agents for maze {maze_number} generated and saved successfully.')
    return a, b


def generate_lmdp_agent(maze_number, agent_number,
                        steps_distribution, ITI_steps_distribution, navigation_time_at_node_distribution,
                        ITI_time_at_node_distribution, reward_time_at_node_distribution,):
    """
    Generate an LMDP agent for a specific maze and agent number.

    :param root_dir: Root directory for data storage.
    :param maze_number: The maze number for which to generate the agent.
    :param agent_number: The specific agent number to generate.
    :param steps_distribution: Distribution for the number of steps in a trial.
    :type steps_distribution: torch.distributions.Distribution
    :param ITI_steps_distribution: Distribution for the number of ITI steps.
    :type ITI_steps_distribution: torch.distributions.Distribution
    :param navigation_time_at_node_distribution: Distribution for the time spent at each node during a trial.
    :type navigation_time_at_node_distribution: torch.distributions.Distribution
    :param ITI_time_at_node_distribution: Distribution for the time spent during ITI.
    :type ITI_time_at_node_distribution: torch.distributions.Distribution
    :param reward_time_at_node_distribution: Distribution for the time spent at the reward node.
    :type reward_time_at_node_distribution: torch.distributions.Distribution
    :return: A tuple containing agent metadata and a DataFrame with the agent's trajectories.
    1. agent metadata: a dictionary with the agent's parameters and routes.
    2. agent_trajectories_data: a DataFrame with the agent's trajectories
    and associated data such as time, trial phase, reward index, position index, start index
    and action class.
    """
    # agent metadata, hardcoded statistics for agent generation
    # -------------------------------------------------------------------------------
    n_routes = np.random.randint(3, 6)  # randomly choose number of routes
    straight_coef = np.random.rand() * 4 + 1  # randomly choose a straight coefficient
    route_length_mean = np.random.rand() * 5 + 9  # mean route length
    route_length_var = np.random.rand() * 4 + 2
    cognitive_constant = np.random.rand() * 20 + 10  # cognitive constant
    action_cost = np.random.rand() * 0.1 + 0.1
    reward_value = np.random.rand() * 2 + 1  # reward value

    # generate routes
    # ------------------------------------------------------------------------------- # variance of route
    routes, route_length = random_route_generate(maze_number=maze_number,
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
        a = planner.generate_trajectory(start, finish)
        trajectory = [route * 196 + state for state, route in a]  # convert to route index
        n_steps = len(trajectory)
        steps += n_steps

        # sample time for each trial step and for the reward time
        # ------------------------------------------------------------------------------------------------------
        navigation_time = navigation_time_at_node_distribution.sample((n_steps,))
        reward_time = reward_time_at_node_distribution.sample()
        navigation_time[-1] = reward_time
        trial_phase = ['navigation'] * n_steps

        # sample the number of ITI steps. Diffusion with no reverse and preference for forward direction
        # ------------------------------------------------------------------------------------------------------
        n_ITI_steps = ITI_steps_distribution.sample().round().int()
        steps += n_ITI_steps
        s_prev, a_prev = finish, trajectory[-1] % 196 // 49
        s_new = s_prev + (((a_prev % 2) == 0) * 7 + ((a_prev % 2) == 1) * 1) * \
                ((a_prev <= 1) * 1 + (a_prev > 1) * -1)
        iti_steps, iti_times = iti_steps_generate(maze_number=maze_number,
                                                  start_location=s_new,
                                                  previous_action=a_prev,
                                                  n_steps=n_ITI_steps,
                                                  iti_steps_dist=ITI_time_at_node_distribution,
                                                  straight_coef=straight_coef,
                                                  reverse_coef=0.999)
        if n_ITI_steps > 0:
            trial_phase.extend(['ITI'] * n_ITI_steps)
            navigation_time = torch.cat([navigation_time, iti_times])
            trajectory.extend([-196 + i for i in iti_steps])  # allocate route index = -1 n_routes + 2 to ITI

        # store the trajectory data
        # ------------------------------------------------------------------------------------------------------
        trial += 1
        trajectories.extend(trajectory)
        rewards.extend([finish] * len(trajectory))
        starts.extend([start] * len(trajectory))
        times = torch.cat([times, navigation_time])
        trials.extend([trial] * len(trajectory))
        trial_phases.extend(trial_phase)
        s_prev, a_prev = (trajectory[-1] % 196) % 49, (trajectory[-1] % 196) // 49
        start = s_prev + (((a_prev % 2) == 0) * 7 + ((a_prev % 2) == 1) * 1) * \
                ((a_prev <= 1) * 1 + (a_prev > 1) * -1)
        prev_reward = finish


    # data to save
    # -------------------------------------------------------------------------------
    agent_data = {
        'maze_id': maze_number,
        'unique_id': f'lmdp_{maze_number}_{agent_number}',
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
        'unique_id': f'lmdp_{maze_number}_{agent_number}',
        'agent_type': 'lmdp_agents',
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


def iti_steps_generate(maze_number, start_location, previous_action, n_steps, iti_steps_dist, straight_coef=3, reverse_coef=0.999):
    """
    Generate ITI steps for the agent's behaviour.
    This function generates a list of ITI steps based on the specified distribution and coefficients.

    :param maze_number: ID of the maze for which to generate ITI steps.
    :param start_location: Starting location for the ITI steps
    :param previous_action: Previous action taken by the agent, used to determine the next action.
    :type previous_action: int, representing the last action taken by the agent.
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
    action_matrix = torch.tensor(POSSIBLE_LOCATION_ACTION_FOR_MAZE_ID[maze_number]).reshape(4, 49).t()
    iti_steps, ITI_time = [], []
    s_new = start_location
    a_prev = previous_action
    if n_steps > 0:
        ITI_time = iti_steps_dist.sample((n_steps,))
        for _ in range(n_steps):
            a_new = Categorical(action_matrix[s_new] * diffusion[a_prev]).sample()
            iti_steps.append(a_new * 49 + s_new)
            s_prev, a_prev = s_new, a_new
            s_new = s_prev + (((a_prev % 2) == 0) * 7 + ((a_prev % 2) == 1) * 1) * \
                    ((a_prev <= 1) * 1 + (a_prev > 1) * -1)
    return iti_steps, ITI_time


