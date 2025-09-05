import torch
from pathlib import Path
from tqdm import tqdm

from regressionhelper.habit_funcs import generate_habits
from datahelper.fields import COMMON_FIELDS
from datahelper.load_data import load_subject_IDs


if __name__ == "__main__":
    root_dir = Path(__file__).parents[1]
    datasets = COMMON_FIELDS.keys()
    maze_numbers = [1, 2, 3]

    total = 0
    for dataset in datasets:
        for maze_number in maze_numbers:
            subject_IDs = load_subject_IDs(dataset)
            kfolds = 13 if (dataset == 'mice_behaviour') and (maze_number == 1) else 11
            total += len(subject_IDs) * kfolds

    with tqdm(total=total) as pbar:
        for dataset in datasets:
            subject_IDs = load_subject_IDs(dataset)
            habits_all = torch.ones(3, len(subject_IDs), 13, 196) * float('nan')
            for maze_number in maze_numbers:
                kfolds = 13 if (dataset == 'mice_behaviour') and (maze_number == 1) else 11
                for subject_ID in subject_IDs:
                    for kfold in range(kfolds):
                        habits_all[maze_number-1, subject_IDs.index(subject_ID), kfold] = generate_habits(
                            dataset=dataset,
                            maze_number=maze_number,
                            subject_ID=subject_ID,
                            kfold=kfold
                        )
                        pbar.update(1)
            # Save habits for the dataset
            torch.save(habits_all, f'{root_dir}/habits/habits_{dataset}.pt')
    print("All habits generated and saved successfully.")