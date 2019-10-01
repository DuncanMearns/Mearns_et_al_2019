from datasets.lensectomy import experiment
from behaviour_analysis.manage_files import create_folder
import os
import pandas as pd
import numpy as np


if __name__ == "__main__":

    # Paths for saving
    experiment.subdirs['analysis'] = os.path.join(experiment.directory, 'analysis')
    eye_convergence_directory = os.path.join(experiment.subdirs['analysis'], 'eye_convergence')
    output_directory = create_folder(eye_convergence_directory, 'hunt_initiation')

    # Open bouts
    mapped_bouts = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'),
                               index_col=0, dtype={'ID': str, 'video_code': str})

    # Calculate the hunt initiation rate for each fish grouped by condition
    hunt_rate_by_condition = {}
    for condition, IDs in experiment.data.groupby('condition')['ID']:
        condition_bouts = mapped_bouts[mapped_bouts['ID'].isin(IDs)]
        condition_start_rate = []
        for ID, fish_bouts in condition_bouts.groupby('ID'):
            print ID, ':',
            n_start = (fish_bouts['phase'] == 1).sum()
            rate = n_start / 20.  # number of start hunts per minute
            print rate
            condition_start_rate.append(rate)
        hunt_rate_by_condition[condition] = condition_start_rate

    control_rate = hunt_rate_by_condition['control']
    unilateral_rate = hunt_rate_by_condition['right'] + hunt_rate_by_condition['left']
    bilateral_rate = hunt_rate_by_condition['bilateral']

    # Save
    np.save(os.path.join(output_directory, 'control_rate.npy'), np.array(control_rate))
    np.save(os.path.join(output_directory, 'unilateral_rate.npy'), np.array(unilateral_rate))
    np.save(os.path.join(output_directory, 'bilateral_rate.npy'), np.array(bilateral_rate))
