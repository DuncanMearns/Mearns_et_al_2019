from datasets.blumenkohl import experiment as blu
from datasets.lakritz import experiment as lak
from behaviour_analysis.manage_files import create_folder
from behaviour_analysis.miscellaneous import print_heading, Timer
from behaviour_analysis.analysis.stimulus_mapping import BoutStimulusMapper
import os
import pandas as pd


if __name__ == "__main__":

    for experiment in (blu, lak):

        print_heading(os.path.basename(experiment.directory))
        stimulus_map_directory = create_folder(experiment.subdirs['analysis'], 'stimulus_maps')

        mapped_bouts = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'), index_col=0,
                                   dtype={'ID': str, 'video_code': str})

        # Calculate stimulus maps for each fish in parallel
        timer = Timer()
        timer.start()
        mapper = BoutStimulusMapper(mapped_bouts, experiment, stimulus_map_directory, n_threads=20)
        mapper.run()
        total_time = timer.stop()
        print 'Total time:', timer.convert_time(total_time)
