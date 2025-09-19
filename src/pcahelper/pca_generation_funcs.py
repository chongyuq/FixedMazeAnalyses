import torch
import os
from pathlib import Path
from datahelper.load_data import get_data
from pcahelper.pca_exponential_funcs import fit_decay_pca_basis


def generate_PCs(dataset, maze_number, subject_ID, kfold: int=None, history=True, future=True, combine='sum', normalize=True, alpha=0.1, nmf=False):
    """
    Generates principal components for the specified dataset, maze number, and subject ID.
    :param dataset: The dataset to use (e.g., 'mice_behaviour', 'lmdp_agents', etc.).
    :param maze_number: The maze number to filter the data.
    :param subject_ID: The subject ID to filter the data.
    :param kfold: Optional k-fold cross-validation index.
    :return: A DataFrame containing the principal components.
    """
    # Load data
    if kfold is not None:
        query = {"subject_ID": subject_ID, "day_on_maze": {"$nin": [kfold+1, kfold+2]}}
    else:
        query = {"subject_ID": subject_ID}
    data = get_data(dataset=dataset, maze_number=maze_number, query=query)

    # process data
    x = torch.tensor(data['pos_idx'].to_numpy(), dtype=torch.long)
    a = torch.tensor(data['action_class'].to_numpy(), dtype=torch.long)
    xa = a * 49 + x  # 49 actions per position
    start_idx = torch.tensor((data.day_on_maze != data.day_on_maze.shift(1)).to_numpy()).nonzero().flatten() # treat each day/session as separate sequences
    time = torch.tensor(data['time'].to_numpy(), dtype=torch.float32)

    # Fit PCA
    pcs, explained_variance = fit_decay_pca_basis(
        idx=xa,
        start_idx=start_idx,
        tot_obs=196,
        t=time,
        alpha=alpha,
        normalize=normalize,
        history=history,
        future=future,
        combine=combine,
        nmf=nmf
    )
    return pcs, explained_variance


def generate_and_save_PCs(dataset, maze_number, subject_ID, kfold: int=None, history=True, future=True, combine='sum', normalize=True, alpha=0.1, nmf=False):
    """
    Saves the principal components to a specified directory.
    :param pcs: The principal components to save.
    :param explained_variance: The explained variance of the principal components.
    :param dataset: The dataset used to generate the PCs.
    :param maze_number: The maze number used to filter the data.
    :param subject_ID: The subject ID used to filter the data.
    :param kfold: Optional k-fold cross-validation index.
    """
    pcs, explained_variance = generate_PCs(dataset, maze_number, subject_ID, kfold, history, future, combine, normalize, alpha, nmf=nmf)
    root_dir = Path(__file__).parents[2]
    if nmf:
        base_dir = f'{root_dir}/inferred_routes/nmf_inferred/history_{history}_future_{future}_combine_{combine}_normalize_{normalize}_alpha_{alpha}/{dataset}/{subject_ID}'
    else:
        base_dir = f'{root_dir}/inferred_routes/pca_inferred/history_{history}_future_{future}_combine_{combine}_normalize_{normalize}_alpha_{alpha}/{dataset}/{subject_ID}'
    os.makedirs(base_dir, exist_ok=True)
    kfold_suffix = f"_kfold_{kfold}" if kfold is not None else ""
    PC_suffix = "_NMFs" if nmf else "_PCs"
    component_name = 'nmfs' if nmf else 'pcs'
    filename = f"{base_dir}/Maze{maze_number}{PC_suffix}{kfold_suffix}.pt"
    torch.save({component_name: pcs, 'explained_variance': explained_variance}, filename)
    print(f'Saved {PC_suffix[1:]} to {filename}')
    return