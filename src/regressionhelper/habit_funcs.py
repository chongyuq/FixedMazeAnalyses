from datahelper.load_data import *
import torch
import numpy as np
import torch.nn.functional as F


def generate_habits(dataset, maze_number, subject_ID, kfold: int=None):
    if kfold is not None:
        query = {"subject_ID": subject_ID, "day_on_maze": {"$nin": [kfold+1, kfold+2]}}
    else:
        query = {"subject_ID": subject_ID}
    data = get_data(dataset=dataset, maze_number=maze_number, query=query)

    # process data
    x = torch.tensor(data['pos_idx'].to_numpy(), dtype=torch.long)
    a = torch.tensor(data['action_class'].to_numpy(), dtype=torch.long)
    xa = a * 49 + x  # 49 actions per position
    habit = F.one_hot(xa, 196).sum(dim=0)
    return habit