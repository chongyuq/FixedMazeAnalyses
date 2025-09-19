from pcahelper.pca_generation_funcs import generate_and_save_PCs
from datahelper.fields import COMMON_FIELDS
from datahelper.load_data import *
from tqdm import tqdm


if __name__ == "__main__":
    agent_type = 'non_markovian_agents'
    maze_numbers = [1, 2, 3]

    # Compute total steps for tqdm
    total = 0
    for maze_number in maze_numbers:
        subject_IDs = load_subject_IDs(agent_type)
        kfolds = 13 if (agent_type == 'mice_behaviour') and (maze_number == 1) else 11
        total += len(subject_IDs) * (kfolds + 1)

    with tqdm(total=total) as pbar:
        for maze_number in maze_numbers:
            subject_IDs = load_subject_IDs(agent_type)
            kfolds = 13 if (agent_type == 'mice_behaviour') and (maze_number == 1) else 11
            for subject_ID in subject_IDs:
                generate_and_save_PCs(
                    dataset=agent_type,
                    maze_number=maze_number,
                    subject_ID=subject_ID,
                    kfold=None,
                    history=True,
                    future=True,
                    combine='sum',
                    normalize=True,
                    alpha=0.1
                )
                pbar.update(1)
                for kfold in range(kfolds):
                    generate_and_save_PCs(
                        dataset=agent_type,
                        maze_number=maze_number,
                        subject_ID=subject_ID,
                        kfold=kfold,
                        history=True,
                        future=True,
                        combine='sum',
                        normalize=True,
                        alpha=0.1
                    )
                    pbar.update(1)

    print(f"PCs for {agent_type} generated and saved successfully.")
