import torch
import numpy as np
from pathlib import Path
import os

from datahelper.load_data import get_data
import pandas as pd


def get_time():
    maze_behaviour = get_data(dataset='mice_behaviour', maze_number=None)
    time_at_node = maze_behaviour.time.shift(-1).to_numpy() - maze_behaviour.time.to_numpy()
    time_at_node[time_at_node < 0] = float('nan')
    maze_behaviour['time_at_node'] = time_at_node
    subject_IDs = maze_behaviour.subject_ID.unique().tolist()
    n = len(subject_IDs)
    rows = []

    # Loop through mazes and subjects
    for maze in range(1, 4):
        for i, subject_ID in enumerate(subject_IDs):
            m = maze_behaviour[
                (maze_behaviour.subject_ID == subject_ID) &
                (maze_behaviour.maze_number == maze)
                ]

            # Select trial phases
            r = m[(m.trial != m.trial.shift(-1)) & (m.trial_phase == 'navigation')]
            t = m[(m.trial == m.trial.shift(-1)) & (m.trial_phase == 'navigation')]
            iti = m[m.trial_phase == 'ITI']

            # Times
            r_time = r.time_at_node.to_numpy()
            t_time = t.time_at_node.to_numpy()
            iti_time = iti.time_at_node.to_numpy()

            # Log times
            log_r_time = np.log(r_time)
            log_t_time = np.log(t_time)
            log_iti_time = np.log(iti_time)

            # Dictionary to hold row
            row = {
                'subject_ID': subject_ID,
                'maze_number': maze,
            }

            # Raw means and stds
            row['reward_time_mean'] = np.nanmean(r_time)
            row['navigation_time_mean'] = np.nanmean(t_time)
            row['iti_time_mean'] = np.nanmean(iti_time)

            row['reward_std'] = np.nanstd(r_time)
            row['navigation_std'] = np.nanstd(t_time)
            row['iti_std'] = np.nanstd(iti_time)

            # Log means and stds
            row['reward_log_mean'] = np.nanmean(log_r_time)
            row['navigation_log_mean'] = np.nanmean(log_t_time)
            row['iti_log_mean'] = np.nanmean(log_iti_time)

            row['reward_log_std'] = np.nanstd(log_r_time)
            row['navigation_log_std'] = np.nanstd(log_t_time)
            row['iti_log_std'] = np.nanstd(log_iti_time)

            rows.append(row)

        # Create the final DataFrame
        df = pd.DataFrame(rows)
    return df


def get_trial_info():
    # trial length
    # session length
    # iti length
    maze_behaviour = get_data(dataset='mice_behaviour', maze_number=None)
    phase_end = maze_behaviour.trial_phase != maze_behaviour.trial_phase.shift(-1)
    maze_behaviour['phase_end'] = phase_end
    subject_IDs = maze_behaviour.subject_ID.unique().tolist()
    n = len(subject_IDs)
    rows = []

    for maze in range(1, 4):
        for subject_ID in subject_IDs:
            m = maze_behaviour[
                (maze_behaviour.subject_ID == subject_ID) &
                (maze_behaviour.maze_number == maze)
                ]

            # Trial end detection
            trial_end = torch.tensor((m.trial != m.trial.shift(-1)).to_numpy())
            trial_end_ind = torch.nonzero(trial_end).flatten()
            trial_length = trial_end_ind - torch.cat([torch.zeros(1, dtype=torch.long), trial_end_ind[:-1]])

            # Session end detection
            session_end = torch.tensor((m.day_on_maze != m.day_on_maze.shift(-1)).to_numpy())
            session_end_ind = torch.nonzero(session_end).flatten()
            session_length = session_end_ind - torch.cat([torch.zeros(1, dtype=torch.long), session_end_ind[:-1]])

            # ITI detection
            ITI_end = torch.tensor(m[m.trial_phase == 'ITI'].phase_end.to_numpy())
            ITI_end_ind = torch.nonzero(ITI_end).flatten()
            ITI_length = ITI_end_ind - torch.cat([torch.zeros(1, dtype=torch.long), ITI_end_ind[:-1]])

            # Store row data
            row = {
                'subject_ID': subject_ID,
                'maze_number': maze,
                'trial_steps_mean': trial_length.float().mean().item(),
                'trial_steps_std': trial_length.float().std().item(),
                'session_steps_mean': session_length.float().mean().item(),
                'session_steps_std': session_length.float().std().item(),
                'ITI_steps_mean': ITI_length.float().mean().item(),
                'ITI_steps_std': ITI_length.float().std().item(),
                'total_steps': len(m)
            }

            rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df_time = get_time()
    df_steps = get_trial_info()
    root_dir = Path(__file__).parents[1]

    # Save DataFrames to CSV files
    os.makedirs(f'{root_dir}/data/behaviour/summary_statistics', exist_ok=True)
    df_time.to_csv(f'{root_dir}/data/behaviour/summary_statistics/time_at_node.csv', index=False)
    df_steps.to_csv(f'{root_dir}/data/behaviour/summary_statistics/trial_session_ITI_steps.csv', index=False)
