import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import os
from datahelper.fields import COMMON_FIELDS
from typing import Optional, Dict
import torch


load_dotenv()


uri = os.getenv("MONGODB_URI")
db_str = os.getenv("MONGODB_DB")
client = MongoClient(uri)
db = client[db_str]


def get_data(
        dataset: str = "mice_behaviour",
        maze_number: int = 1,
        query: Optional[Dict] = None
):
    """
    Fetches data from the MongoDB database.
    :param dataset: Type of agent data to fetch, e.g., 'mice_behaviour', 'lmdp_agents', 'dijkstra_agents', 'non_markovian_agents', 'markovian_agents'.
    :param maze_number: The ID of the maze to filter the data.
    :param query: Additional query parameters to filter the data.
    :return: A DataFrame containing the fetched data.
    """
    if query is None:
        query = {}
    if maze_number is not None:
        assert maze_number > 0 and maze_number < 4, "Maze number must be between 1 and 3."
    collection = COMMON_FIELDS[dataset]['collection']
    fields = COMMON_FIELDS[dataset]['fields']
    sort_fields = COMMON_FIELDS[dataset]['sort']
    if maze_number is None:
        query = query
    else:
        query = {'maze_number': maze_number, **query}
    dataframe = pd.DataFrame(list(db[collection].find(query, fields).sort(sort_fields)))
    if dataset != 'mice_behaviour':
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
    return np.load(f"{Path(__file__).parents[2]}/data/synthetic_data/optimal/maze{maze_number}.npy")


def load_all_kfold_habits_for_dataset(dataset: str = "mice_behaviour"):
    """
    Loads habit data from pre-saved files.
    :param dataset: The dataset type, e.g., 'behaviour', 'lmdp_agents', etc.
    :return: A tensor containing the habit data. The shape is (3, num_subjects, num_days, 196).
    the first dimension corresponds to the maze number (1, 2, 3).
    the second dimension corresponds to different subjects, ordered by load_subject_IDs().
    the third dimension corresponds to different days on the maze - this corresponds to different kfolds. If it's nan, it means that fold doesn't exist.
    So 5 in the third dimension means the data from day 6 and day 7 were left out during habit calculation.
    the last dimension corresponds to the 196 possible (position, action) pairs.
    196 = 49 positions * 4 actions.
    49 positions correspond to a 7x7 grid world.
    4 actions correspond to up, down, left, right.
    Note that the habit is not normalized, so it is just a count of how many times each (position, action) pair was taken.
    """
    assert dataset in COMMON_FIELDS, f"Dataset {dataset} is not recognized."
    root_dir = Path(__file__).parents[2]
    return torch.load(f"{root_dir}/habits/habits_{dataset}.pt")


def load_all_kfold_pcs_for_dataset(dataset: str = "mice_behaviour", history=True, future=True, combine='sum', normalize=True, alpha=0.1):
    """
    Loads all precomputed principal components (PCs) for a given dataset.
    :param dataset: The dataset type, e.g., 'mice_behaviour', 'lmdp_agents', etc.
    :param history: Whether to include historical data in the PCs.
    :param future: Whether to include future data in the PCs.
    :param combine: Method to combine history and future ('sum' or 'concat').
    :param normalize: Whether to normalize the data before PCA.
    :param alpha: Decay parameter for exponential weighting.
    :return: A dictionary containing:
        - 'variances': A tensor of shape (3, num_subjects, 13) containing the explained variances for each maze, subject, and kfold.
        - 'PCs': A tensor of shape (3, num_subjects, 13, 196, 196) containing the principal components for each maze, subject, and kfold.
    The first dimension corresponds to the maze number (1, 2, 3).
    The second dimension corresponds to different subjects, ordered by load_subject_IDs().
    The third dimension corresponds to different kfolds (0-12).
    The fourth dimension corresponds to the 196 possible (position, action) pairs.
    The fifth dimension corresponds to the 196 principal components.
    196 = 49 positions * 4 actions.
    49 positions correspond to a 7x7 grid world.
    4 actions correspond to up, down, left, right.
    """
    assert dataset in COMMON_FIELDS, f"Dataset {dataset} is not recognized."
    root_dir = Path(__file__).parents[2]
    base_dir = f'{root_dir}/inferred_routes/pca_inferred/history_{history}_future_{future}_combine_{combine}_normalize_{normalize}_alpha_{alpha}/{dataset}'
    variances, PCs = torch.ones(3, 6, 13, 196), torch.zeros(3, 6, 13, 196, 196)  # (maze_number, subject_ID, kfold, n_obs=196, n_pcs=196)
    subject_IDs = load_subject_IDs(dataset)
    for subject_ID in subject_IDs:
        for maze_number in range(1, 4):
            for kfold in range(13):
                filename = f'{base_dir}/{subject_ID}/Maze{maze_number}_PCs_kfold_{kfold}.pt'
                if os.path.exists(filename):
                    pcs = torch.load(filename)
                    PCs[maze_number - 1, subject_IDs.index(subject_ID), kfold] = pcs['pcs']
                    variances[maze_number - 1, subject_IDs.index(subject_ID), kfold] = pcs['explained_variance']
    return {'variances': variances, 'PCs': PCs}


def load_all_kfold_nmfs_for_dataset(dataset: str = "mice_behaviour", history=True, future=True, combine='sum', normalize=True, alpha=0.1):
    """
    Loads all precomputed principal components (PCs) for a given dataset.
    :param dataset: The dataset type, e.g., 'mice_behaviour', 'lmdp_agents', etc.
    :param history: Whether to include historical data in the PCs.
    :param future: Whether to include future data in the PCs.
    :param combine: Method to combine history and future ('sum' or 'concat').
    :param normalize: Whether to normalize the data before PCA.
    :param alpha: Decay parameter for exponential weighting.
    :return: A dictionary containing:
        - 'variances': A tensor of shape (3, num_subjects, 13) containing the explained variances for each maze, subject, and kfold.
        - 'PCs': A tensor of shape (3, num_subjects, 13, 196, 196) containing the principal components for each maze, subject, and kfold.
    The first dimension corresponds to the maze number (1, 2, 3).
    The second dimension corresponds to different subjects, ordered by load_subject_IDs().
    The third dimension corresponds to different kfolds (0-12).
    The fourth dimension corresponds to the 196 possible (position, action) pairs.
    The fifth dimension corresponds to the 196 principal components.
    196 = 49 positions * 4 actions.
    49 positions correspond to a 7x7 grid world.
    4 actions correspond to up, down, left, right.
    """
    assert dataset in COMMON_FIELDS, f"Dataset {dataset} is not recognized."
    root_dir = Path(__file__).parents[2]
    base_dir = f'{root_dir}/inferred_routes/nmf_inferred/history_{history}_future_{future}_combine_{combine}_normalize_{normalize}_alpha_{alpha}/{dataset}'
    variances, NMFs = torch.ones(3, 6, 13, 196), torch.zeros(3, 6, 13, 196, 6)  # (maze_number, subject_ID, kfold, n_obs=196, n_pcs=6)
    subject_IDs = load_subject_IDs(dataset)
    for subject_ID in subject_IDs:
        for maze_number in range(1, 4):
            for kfold in range(13):
                filename = f'{base_dir}/{subject_ID}/Maze{maze_number}_NMFs_kfold_{kfold}.pt'
                if os.path.exists(filename):
                    nmfs = torch.load(filename)
                    NMFs[maze_number - 1, subject_IDs.index(subject_ID), kfold] = nmfs['nmfs']
                    # variances[maze_number - 1, subject_IDs.index(subject_ID), kfold] = pcs['explained_variance']
    return {'variances': None, 'NMFs': NMFs}


def load_pcs(dataset: str = "mice_behaviour", subject_ID: str = "m2", maze_number: int = 1, kfold: Optional[int] = None, history=True, future=True, combine='sum', normalize=True, alpha=0.1):
    """
    Loads precomputed principal components (PCs) for a given dataset, subject ID, and maze number.
    :param dataset: The dataset type, e.g., 'mice_behaviour', 'lmdp_agents', etc.
    :param subject_ID: The subject ID to load PCs for.
    :param maze_number: The maze number to load PCs for.
    :param kfold: Optional k-fold cross-validation index.
    :param history: Whether to include historical data in the PCs.
    :param future: Whether to include future data in the PCs.
    :param combine: Method to combine history and future ('sum' or 'concat').
    :param normalize: Whether to normalize the data before PCA.
    :param alpha: Decay parameter for exponential weighting.
    :return: A tensor containing the principal components.
    """
    assert dataset in COMMON_FIELDS, f"Dataset {dataset} is not recognized."
    assert maze_number > 0 and maze_number < 4, "Maze number must be between 1 and 3."
    root_dir = Path(__file__).parents[2]
    base_dir = f'{root_dir}/inferred_routes/pca_inferred/history_{history}_future_{future}_combine_{combine}_normalize_{normalize}_alpha_{alpha}/{dataset}/{subject_ID}'
    if kfold is not None:
        filename = f'{base_dir}/Maze{maze_number}_PCs_kfold_{kfold}.pt'
    else:
        filename = f'{base_dir}/Maze{maze_number}_PCs.pt'
    return torch.load(filename)


def load_subject_IDs(agent_type: str = "mice_behaviour") -> list:
    """
    Fetches unique subject IDs from the database for a given agent type.
    :param agent_type: Type of agent data to fetch subject IDs for.
    :return: A list of unique subject IDs.
    """
    assert agent_type in COMMON_FIELDS, f"Agent type {agent_type} is not recognized."
    collection = COMMON_FIELDS[agent_type]['collection']
    return sorted(list(db[collection].distinct('subject_ID')))




