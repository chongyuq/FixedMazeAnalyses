from nonmarkovianhelper.nonmarkovian_building_funcs import NonMarkovianTrainingDataBuilder, SingleActionUniquePredictabilityFinder
from regressionhelper.regressor_building_funcs import UniquePredictabilityFinder
import os
from pathlib import Path
from tqdm import tqdm
import torch


def run_nonmarkovian_pipeline(
    root_dir: str,
    maze_number: int,
    dataset: str,
    tot_steps_back: int = 4,
    save_model: bool = True
):
    """
    Main regression pipeline to generate synthetic agents and build a regression model.

    Args:
        root_dir (str): The root directory for saving models and data.
        maze_number (int): The maze number to use for the regression.
        dataset (str): The dataset to use, e.g., 'mice_behaviour', 'lmdp_agents', etc.
        save_model (bool, optional): Whether to save the trained model. Defaults to True.

    Returns: If not saving, it will return a dictionary containing:
        - 'coefs': Coefficients of the regression model for each fold (or day on maze) for each bootstrap (or subject) for all the models.
        The shape is (n_folds/n_days, n_bootstraps/n_subjects, n_models/n_regressors + 1, n_regressors).
        - 'neg_log_likelihoods': Negative log likelihoods for each fold (or day on maze) for each bootstrap (or subject) for all the models.
        The shape is (n_folds/n_days, n_bootstraps/n_subjects, n_models/n_regressors + 1).
        - 'accuracies': Accuracies for each fold (or day on maze) for each bootstrap (or subject) for all the models.
        The shape is (n_folds/n_days, n_bootstraps/n_subjects, n_models/n_regressors + 1).
        None
    """
    # Step 1: build the regression model and get features for training and validation
    DataBuilder = NonMarkovianTrainingDataBuilder(
        dataset = dataset,
        maze_number = maze_number,
        tot_steps_back= tot_steps_back
    )

    X_train, Y_train, X_validation, Y_validation = DataBuilder.build()

    # Step 2: Find the unique predictability of each regressor
    UniquePredictability = SingleActionUniquePredictabilityFinder(
        regressors=['T', 'a_1', 'a_2', 'a_3', 'a_4'],  # time step and actions
    )

    # UniquePredictability = UniquePredictabilityFinder(
    #     regressors=['T', 'a_1', 'a_2', 'a_3', 'a_4'],  # time step and actions
    # )

    kfolds = len(X_train)
    subject_IDs = list(range(kfolds))
    x = UniquePredictability.bootstrap_or_regress_per_subject(
        subject_IDs=subject_IDs,
        X_t=[X_train[k][0] for k in range(kfolds)],
        Y_t=[Y_train[k][0] for k in range(kfolds)],
        X_v=[X_validation[k][0] for k in range(kfolds)],
        Y_v=[Y_validation[k][0] for k in range(kfolds)],
        bootstraps=None,
    )

    if save_model:
        save_path = f"{root_dir}/regression_results/is_it_markovian_results/{dataset}"
        os.makedirs(save_path, exist_ok=True)
        torch.save({
            'coefs': x['coefs'],
            'neg_log_likelihoods': x['neg_log_likelihoods'],
            'accuracies': x['accuracies']
        }, f"{save_path}/maze_{maze_number}_regression_model.pt")
        print(f"Model saved to {save_path}/maze_{maze_number}_regression_model.pt")
        return
    else:
        return x


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    for dataset in ['mice_behaviour', 'lmdp_agents', 'dijkstra_agents', 'non_markovian_agents', 'markovian_agents']:
        test = run_nonmarkovian_pipeline(
            root_dir = Path(__file__).parents[2],
            maze_number = 1,
            dataset = dataset,
            tot_steps_back = 4,
            save_model = False
        )

    # #     import matplotlib.pyplot as plt
    #     print(test['coefs'].shape)  # (n_folds, n_bootstraps, n_models, n_regressors)
    #     print(test['neg_log_likelihoods'].shape) # (n_folds, n_bootstraps, n_models)
    #     print(test['accuracies'].shape) # (n_folds, n_bootstraps, n_models)

        x = test['neg_log_likelihoods'][:, :-1] - test['neg_log_likelihoods'][:, -1:]
        plt.plot(x.mean(dim=0)[2:])
        plt.title(f'{dataset}')
        plt.show()

