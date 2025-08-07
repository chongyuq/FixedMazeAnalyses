import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from mazehelper.mazes_info import POSSIBLE_LOCATION_ACTION_FOR_MAZE_ID


def plot_maze(maze_id,
              ax=None,
              fig=None,
              color='grey',
              transition_linewidth=6,
              node_width=200):
    """
    Plot the maze with the specified maze_id.
    :param maze_id: maze identifier, should be between 1 and 3
    :type maze_id: int
    :param ax: matplotlib axis to plot on, if None, current axis is used
    :type ax:
    :param fig: matplotlib figure to plot on, if None, current figure is used
    :type fig:
    :param color: color of the bridges and nodes
    :type color: str
    :param transition_linewidth: bridge line width
    :type transition_linewidth: int
    :param node_width: width of the nodes in the maze
    :type node_width: int
    """
    transition_policy = torch.tensor(POSSIBLE_LOCATION_ACTION_FOR_MAZE_ID[maze_id]).reshape(4, 49).t().reshape(7, 7, 4)
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()
    for i in range(transition_policy.shape[0]):
        for j in range(transition_policy.shape[1]):
            ax.arrow(i, j, transition_policy[i, j, 0], 0, color=color, zorder=0, linewidth=transition_linewidth, head_width=0)  # linestyle=(0, (1, 10)),
            ax.arrow(i, j, -transition_policy[i, j, 2], 0, color=color, zorder=0, linewidth=transition_linewidth, head_width=0)  # linestyle=(0, (1, 10)),
            ax.arrow(i, j, 0, transition_policy[i, j, 1], color=color, zorder=0, linewidth=transition_linewidth, head_width=0)  # linestyle=(0, (1, 10)),
            ax.arrow(i, j, 0, -transition_policy[i, j, 3], color=color, zorder=0, linewidth=transition_linewidth, head_width=0)  # linestyle=(0, (1, 10)),
    ax.scatter(np.arange(7).repeat(7), np.array(list(range(7)) * 7), color=color, zorder=0, marker="8", s=node_width)  #200
    return


def plot_policy_all(policy,
                    maze_id,
                    ax=None,
                    plot_maze_kwargs=None,
                    **kwargs):
    """
    plot_policy_all plots the policy on the maze.
    :param policy: policy tensor of shape (7, 7, 4) where the last dimension represents the actions (right, up, left, down).
    :type policy: torch.Tensor
    :param maze_id: maze identifier, should be between 1 and 3
    :type maze_id: int
    :param ax: matplotlib axis to plot on, if None, current axis is used
    :type ax:
    :param plot_maze_kwargs: kwargs for the plot_maze function
    :type plot_maze_kwargs:
    :param kwargs: quiver kwargs, such as color, scale, etc.
    :type kwargs:
    :return:
    :rtype:
    """
    if ax is None:
        ax = plt.gca()
    ax.quiver(torch.arange(7).unsqueeze(-1).repeat(1, 7).flatten(),
              torch.arange(7).repeat(7),
              policy[..., 0],
              0, **kwargs)
    ax.quiver(torch.arange(7).unsqueeze(-1).repeat(1, 7).flatten(),
              torch.arange(7).repeat(7),
              -policy[..., 2],
              0, **kwargs)
    ax.quiver(torch.arange(7).unsqueeze(-1).repeat(1, 7).flatten(),
              torch.arange(7).repeat(7),
              0, policy[..., 1],
              **kwargs)
    ax.quiver(torch.arange(7).unsqueeze(-1).repeat(1, 7).flatten(),
              torch.arange(7).repeat(7),
              0,
              -policy[..., 3], **kwargs)
    if plot_maze_kwargs is not None:
        plot_maze(maze_id, ax=ax, **plot_maze_kwargs)
    else:
        plot_maze(maze_id, ax=ax, color='gainsboro')
    ax.set_aspect('equal')
    return


def plot_policy_max(policy, maze_id, ax=None, fig=None, cmap=None, plot_maze_kwargs=None, **kwargs):
    """

    :param policy: shape 49 x 4
    :type policy:
    :param maze_id: id of maze
    :type maze_id:
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:
    """
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    best_actions = policy.max(dim=-1)[1]
    best_actions = F.one_hot(best_actions, 4)
    ax.imshow(policy.reshape(7, 7, 4).transpose(0, 1).max(dim=-1)[0], cmap=cmap)
    ax.quiver(torch.arange(7).unsqueeze(-1).repeat(1, 7).flatten(),
              torch.arange(7).repeat(7),
              (best_actions[..., 0] - best_actions[..., 2]) * policy.max(dim=-1)[0],
              (best_actions[..., 1] - best_actions[..., 3]) * policy.max(dim=-1)[0],
              **kwargs)
    if plot_maze_kwargs is not None:
        plot_maze(maze_id, ax=ax, **plot_maze_kwargs)
    else:
        plot_maze(maze_id, ax=ax)
    ax.invert_yaxis()
    return