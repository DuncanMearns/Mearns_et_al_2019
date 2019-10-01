from datasets.lensectomy import experiment
from paths import paths, capture_strike_directory
from behaviour_analysis.analysis.stimulus_mapping import calculate_fish_sequences
from behaviour_analysis.manage_files import create_folder
from behaviour_analysis.miscellaneous import Timer
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import os


hunting_sequence_directory = create_folder(capture_strike_directory, 'hunting_sequences')


if __name__ == "__main__":

    mapped_bouts = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'),
                               index_col=0, dtype={'ID': str, 'video_code': str})
    capture_strikes = pd.read_csv(paths['capture_strikes'], index_col=0, dtype={'ID': str, 'video_code': str})
    mapped_bouts['strike_label'] = -1
    mapped_bouts.loc[capture_strikes.index, 'strike_label'] = capture_strikes['cluster_label']

    # Take final bouts of hunting sequences
    end_hunts = mapped_bouts[mapped_bouts['phase'] == 3]
    end_hunts = end_hunts[(end_hunts['start'] - 500) >= 0]
    end_hunts.to_csv(os.path.join(hunting_sequence_directory, 'hunting_sequences.csv'), index=True)

    analysis_times = Parallel(4)(delayed(calculate_fish_sequences)(ID, end_hunts, experiment, hunting_sequence_directory)
                                 for ID in end_hunts['ID'].unique())
    print 'Total time:', Timer.convert_time(np.sum(analysis_times))
    print 'Average time:', Timer.convert_time(np.mean(analysis_times))
