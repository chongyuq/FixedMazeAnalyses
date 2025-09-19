from datahelper.fields import COMMON_FIELDS
from datahelper.load_data import *


if __name__ == "__main__":
    datasets = COMMON_FIELDS.keys()
    maze_numbers = [1, 2, 3]
    df_models = []
    df_hpo = []

    for dataset in datasets:
        for maze_number in maze_numbers:
            kfolds = 13 if (dataset == 'mice_behaviour') and (maze_number == 1) else 11

            for subject_index in range(6):
                df_hpo.append({
                    'dataset': dataset,
                    'maze_number': maze_number,
                    'subject_index': subject_index
                })
                for kfold in range(kfolds + 1):
                    df_models.append({
                        'dataset': dataset,
                        'maze_number': maze_number,
                        'subject_index': subject_index,
                        'kfold': int(kfold) if kfold < kfolds else None
                    })

    df_models = pd.DataFrame(df_models)
    df_hpo = pd.DataFrame(df_hpo)

    root_dir = Path(__file__).parents[2]
    os.makedirs(f"{root_dir}/inferred_routes/lowrank_lmdp_inferred/configs", exist_ok=True)
    df_models.to_csv(f"{root_dir}/inferred_routes/lowrank_lmdp_inferred/configs/models_to_run.txt", index=False, sep='\t')
    df_hpo.to_csv(f"{root_dir}/inferred_routes/lowrank_lmdp_inferred/configs/hpo_configs.txt", index=False, sep='\t')
    print(f"CSV files generated and saved to {root_dir}/inferred_routes/hpo_configs and {root_dir}/inferred_routes/models_to_run")
