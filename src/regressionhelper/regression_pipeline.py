from regressionhelper.regressor_building_funcs import *
import os
from pathlib import Path
from tqdm import tqdm


def run_regression_pipeline(
    root_dir: str,
    maze_number: int,
    dataset: str,
    regressors: list,
    bootstraps: int = None,
    save_model: bool = True
):
    """
    Main regression pipeline to generate synthetic agents and build a regression model.

    Args:
        root_dir (str): The root directory for saving models and data.
        maze_number (int): The maze number to use for the regression.
        dataset (str): The dataset to use, e.g., 'mice_behaviour', 'lmdp_agents', etc.
        regressors (list): List of regressors to include in the model. Note that the order matters,
        please use the order found in regressionhelper.constants.REGRESSORS.
        bootstraps (int, optional): Number of bootstrap samples to use. Defaults to None, this then does subjectwise regression.
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
    DataBuilder = TrainingDataBuilder(
        dataset = dataset,
        maze_number = maze_number,
        regressors = regressors
    )

    X_train, Y_train, X_validation, Y_validation = DataBuilder.build()

    # Step 2: Find the unique predictability of each regressor
    UniquePredictability = UniquePredictabilityFinder(
        regressors=regressors,
    )

    coefs, neg_log_likelihoods, accuracies = [], [], []
    subject_IDs = load_subject_IDs(dataset)
    kfolds = len(X_train)
    for kfold in tqdm(range(kfolds), desc="running regression on folds/days"):
         x = UniquePredictability.bootstrap_or_regress_per_subject(
             subject_IDs=subject_IDs,
             X_t=X_train[kfold],
             Y_t=Y_train[kfold],
             X_v=X_validation[kfold],
             Y_v=Y_validation[kfold],
             bootstraps=bootstraps,
         )
         coefs.append(x['coefs'])
         neg_log_likelihoods.append(x['neg_log_likelihoods'])
         accuracies.append(x['accuracies'])

    if save_model:
        aliases = [REGRESSOR_ALIASES[r] for r in regressors]
        folder_name = '_'.join(aliases)
        save_path = f"{root_dir}/regression_results/is_it_using_routes_results/{dataset}/{folder_name}"
        os.makedirs(save_path, exist_ok=True)
        torch.save({
            'coefs': torch.stack(coefs),
            'neg_log_likelihoods': torch.stack(neg_log_likelihoods),
            'accuracies': torch.stack(accuracies)
        }, f"{save_path}/maze_{maze_number}_regression_model.pt")
        print(f"Model saved to {save_path}/maze_{maze_number}_regression_model.pt")
        return
    else:
        return {
            'coefs': torch.stack(coefs),
            'neg_log_likelihoods': torch.stack(neg_log_likelihoods),
            'accuracies': torch.stack(accuracies)
        }


# if __name__ == "__main__":
#     # Example usage
#     test = run_regression_pipeline(
#         root_dir = Path(__file__).parents[2],
#         maze_number = 1,
#         dataset = "mice_behaviour",
#         regressors = ["vector", "optimal", "forward", "reverse"],
#         bootstraps = None,
#         save_model = False
#     )
#
#     import matplotlib.pyplot as plt
#     print(test['coefs'].shape)  # (n_folds, n_bootstraps, n_models, n_regressors)
#     print(test['neg_log_likelihoods'].shape) # (n_folds, n_bootstraps, n_models)
#     print(test['accuracies'].shape) # (n_folds, n_bootstraps, n_models)



