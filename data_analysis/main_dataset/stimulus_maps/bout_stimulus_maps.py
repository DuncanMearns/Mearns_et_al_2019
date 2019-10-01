from datasets.main_dataset import experiment
from paths import stimulus_map_directory
from behaviour_analysis.analysis.stimulus_mapping import BoutStimulusMapper
from behaviour_analysis.miscellaneous import Timer
import os
import pandas as pd


if __name__ == "__main__":

    mapped_bouts = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'), index_col=0,
                               dtype={'ID': str, 'video_code': str})

    # Calculate stimulus maps for each fish in parallel
    timer = Timer()
    timer.start()
    mapper = BoutStimulusMapper(mapped_bouts, experiment, stimulus_map_directory, n_threads=15)
    mapper.run()
    total_time = timer.stop()
    print 'Total time:', timer.convert_time(total_time)
