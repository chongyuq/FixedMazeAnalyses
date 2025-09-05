from pcahelper.pca_generation_funcs import generate_and_save_PCs
from datahelper.fields import COMMON_FIELDS
from datahelper.load_data import *
from tqdm import tqdm


if __name__ == "__main__":
    datasets = COMMON_FIELDS.keys()
    maze_numbers = [1, 2, 3]

    # Compute total steps for tqdm
    total = 0
    for dataset in datasets:
        for maze_number in maze_numbers:
            subject_IDs = load_subject_IDs(dataset)
            kfolds = 13 if (dataset == 'mice_behaviour') and (maze_number == 1) else 11
            total += len(subject_IDs) * (kfolds + 1)

    with tqdm(total=total) as pbar:
        for dataset in datasets:
            for maze_number in maze_numbers:
                subject_IDs = load_subject_IDs(dataset)
                kfolds = 13 if (dataset == 'mice_behaviour') and (maze_number == 1) else 11
                for subject_ID in subject_IDs:
                    generate_and_save_PCs(
                        dataset=dataset,
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
                            dataset=dataset,
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

    print("All principal components generated and saved successfully.")
