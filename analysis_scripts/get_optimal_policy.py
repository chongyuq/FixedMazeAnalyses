import numpy as np
import torch
from pathlib import Path
import os
from mazehelper.mazes_info import POSSIBLE_LOCATION_ACTION_FOR_MAZE_ID
from mazehelper.transition_matrix_functions import transition_matrix_from_action_matrix
import heapq
from mazehelper.plotting_functions import plot_policy_all
import matplotlib.pyplot as plt


def build_policy(maze_number):
    """
    Build an optimal policy for the maze using Dijkstra's algorithm.
    The policy is represented as a 3D numpy array where the first dimension
    corresponds to the goal location, the second dimension corresponds to the
    current state, and the third dimension corresponds to the action taken.
    :param maze_number: The maze identifier, should be between 1 and 3.
    :type maze_number: int
    :return: A 3D numpy array representing the optimal policy.
    :rtype: np.ndarray
    """

    # Get the adjacency matrix for the maze
    adj_matrix = transition_matrix_from_action_matrix(
        torch.tensor(POSSIBLE_LOCATION_ACTION_FOR_MAZE_ID[maze_number]).reshape(
            4, 49).t(), ny=7, nx=7).numpy()
    n = adj_matrix.shape[0]  # 49
    policy = np.zeros((n, n, 4), dtype=np.float32)

    # Convert index to (x, y)
    def idx_to_pos(i):
        x = i // 7
        y = i % 7
        return x, y

    # Convert (x, y) to index
    def pos_to_idx(x, y):
        return x * 7 + y

    # Determine action index
    def get_action(from_idx, to_idx):
        fx, fy = idx_to_pos(from_idx)
        tx, ty = idx_to_pos(to_idx)
        dx, dy = tx - fx, ty - fy
        if dx == 0 and dy == 1:
            return 1  # right
        elif dx == -1 and dy == 0:
            return 2  # up
        elif dx == 0 and dy == -1:
            return 3  # left
        elif dx == 1 and dy == 0:
            return 0  # down
        else:
            return None  # invalid move

    # For each goal location
    for goal in range(n):
        # Use Dijkstra's algorithm (unweighted, so it's effectively BFS)
        dist = [float('inf')] * n
        dist[goal] = 0
        heap = [(0, goal)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v in range(n):
                if adj_matrix[u][v] == 1 and dist[v] > dist[u] + 1:
                    dist[v] = dist[u] + 1
                    heapq.heappush(heap, (dist[v], v))

        # For each state, check which neighbors are closer to the goal
        for state in range(n):
            if state == goal:
                continue  # No action needed

            for neighbor in range(n):
                if adj_matrix[state][neighbor] == 1 and dist[neighbor] < dist[state]:
                    action = get_action(state, neighbor)
                    if action is not None:
                        policy[goal, state, action] = 1.0
            # Normalize the policy to ensure it sums to 1 for each state
            policy[goal, state] /= np.sum(policy[goal, state]) if np.sum(policy[goal, state]) > 0 else 1
    return policy


if __name__ == "__main__":
    root_dir = Path(__file__).parents[1]
    optimal_policy_dir = f'{root_dir}/data/synthetic_data/optimal'
    for maze_number in range(1, 4):
        optimal_policy = build_policy(maze_number)
        # save optimal policy to a file
        np.save(f'{optimal_policy_dir}/maze{maze_number}.npy', optimal_policy)
        # print(f"Optimal policy for maze {maze_number}:\n{optimal_policy}")
        # You can save or visualize the policy as needed
        # For example, you can use matplotlib to visualize the policy
        plot_policy_all(optimal_policy[0], maze_number=maze_number, scale=10)
        plt.show()


