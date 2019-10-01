from datasets.lensectomy import experiment
from paths import paths, capture_strike_directory
from behaviour_analysis.manage_files import create_folder
import pandas as pd
import os
import numpy as np


if __name__ == "__main__":

    mapped_bouts = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'), index_col=0,
                               dtype={'ID': str, 'video_code': str})
    end_hunts = mapped_bouts[(mapped_bouts['phase'] == 3)]
    capture_strikes = pd.read_csv(paths['capture_strikes'], index_col=0, dtype={'ID': str, 'video_code': str})

    output_directory = create_folder(capture_strike_directory, 'proportion_strikes')

    fish_strike_proportions = {}
    for condition, IDs in experiment.data.groupby('condition')['ID']:
        condition_proportions = []
        for idx, ID in IDs.iteritems():
            n_end_hunts = (end_hunts['ID'] == ID).sum()
            n_strikes = (capture_strikes['ID'] == ID).sum()
            if n_end_hunts > 0:
                proportion = n_strikes / float(n_end_hunts)
            else:
                proportion = 0
            condition_proportions.append(proportion)
        fish_strike_proportions[condition] = condition_proportions

    control_proportions = fish_strike_proportions['control']
    unilateral_proportions = fish_strike_proportions['right'] + fish_strike_proportions['left']
    bilateral_proportions = fish_strike_proportions['bilateral']

    # Save
    np.save(os.path.join(output_directory, 'control_proportions.npy'), np.array(control_proportions))
    np.save(os.path.join(output_directory, 'unilateral_proportions.npy'), np.array(unilateral_proportions))
    np.save(os.path.join(output_directory, 'bilateral_proportions.npy'), np.array(bilateral_proportions))
