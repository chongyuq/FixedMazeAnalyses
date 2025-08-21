import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import os
from datahelper.fields import COMMON_FIELDS
from typing import Optional, Dict


load_dotenv()


uri = os.getenv("MONGODB_URI")
db_str = os.getenv("MONGODB_DB")
client = MongoClient(uri)
db = client[db_str]


def get_data(
        agent_type: str = "mice_behaviour",
        maze_number: int = 1,
        query: Optional[Dict] = None
):
    """
    Fetches data from the MongoDB database.
    :param agent_type: Type of agent data to fetch, e.g., 'mice_behaviour', 'lmdp_agents', 'dijkstra_agents', 'non_markovian_agents', 'markovian_agents'.
    :param maze_number: The ID of the maze to filter the data.
    :param query: Additional query parameters to filter the data.
    :return: A DataFrame containing the fetched data.
    """
    if query is None:
        query = {}
    if maze_number is not None:
        assert maze_number > 0 and maze_number < 4, "Maze number must be between 1 and 3."
    collection = COMMON_FIELDS[agent_type]['collection']
    fields = COMMON_FIELDS[agent_type]['fields']
    sort_fields = COMMON_FIELDS[agent_type]['sort']
    if maze_number is None:
        query = query
    else:
        query = {'maze_number': maze_number, **query}
    dataframe = pd.DataFrame(list(db[collection].find(query, fields).sort(sort_fields)))
    if agent_type != 'mice_behaviour':
        # For synthetic agents, we need to calculate the reward angles
        pos, reward = dataframe.pos_idx.to_numpy(), dataframe.reward_idx.to_numpy()
        x, y = pos // 7, pos % 7
        r_x, r_y = reward // 7, reward % 7
        reward_cos_angle = (r_x - x) / ((r_x - x) ** 2 + (r_y - y) ** 2 + 1e-9) ** 0.5
        reward_sin_angle = (r_y - y) / ((r_x - x) ** 2 + (r_y - y) ** 2 + 1e-9) ** 0.5
        dataframe['reward_cos_angle'] = reward_cos_angle
        dataframe['reward_sin_angle'] = reward_sin_angle
    return dataframe


def load_optimal_behaviour(maze_number: int):
    """
    Fetches optimal behaviour data for a specific maze.
    :param maze_number: The ID of the maze to fetch optimal behaviour for.
    :return: A numpy array containing the optimal behaviour data.
    """
    assert maze_number > 0 and maze_number < 4, "Maze number must be between 1 and 3."
    return np.load(f"{Path(__file__).parent[1]}/data/synthetic_data/optimal/maze{maze_number}.npy")


def load_subject_IDs(agent_type: str = "mice_behaviour") -> list:
    """
    Fetches unique subject IDs from the database for a given agent type.
    :param agent_type: Type of agent data to fetch subject IDs for.
    :return: A list of unique subject IDs.
    """
    assert agent_type in COMMON_FIELDS, f"Agent type {agent_type} is not recognized."
    collection = COMMON_FIELDS[agent_type]['collection']
    return list(db[collection].distinct('subject_ID'))




