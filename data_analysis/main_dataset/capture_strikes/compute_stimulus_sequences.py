from datasets.main_dataset import experiment
from paths import paths, capture_strike_directory
from behaviour_analysis.analysis.stimulus_mapping import calculate_fish_sequences
from behaviour_analysis.manage_files import create_folder
from behaviour_analysis.miscellaneous import Timer
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import os


strike_sequence_directory = create_folder(capture_strike_directory, 'strike_sequences')


if __name__ == "__main__":

    capture_strikes = pd.read_csv(paths['capture_strikes'], index_col=0, dtype={'ID': str, 'video_code': str})
    complete_strikes = capture_strikes[(capture_strikes['start'] - 500) >= 0]

    complete_strikes.to_csv(os.path.join(strike_sequence_directory, 'complete_strikes.csv'), index=True)

    analysis_times = Parallel(4)(delayed(calculate_fish_sequences)(ID, complete_strikes, experiment, strike_sequence_directory)
                                 for ID in complete_strikes['ID'].unique())
    print 'Total time:', Timer.convert_time(np.sum(analysis_times))
    print 'Average time:', Timer.convert_time(np.mean(analysis_times))
