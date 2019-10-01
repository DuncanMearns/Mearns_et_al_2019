from datasets.blumenkohl import experiment as blu
from datasets.lakritz import experiment as lak
from behaviour_analysis.analysis.stimulus_mapping import RandomStimulusMapper
from behaviour_analysis.miscellaneous import Timer
import os


if __name__ == "__main__":

    for experiment in (blu, lak):

        stimulus_map_directory = os.path.join(experiment.subdirs['analysis'], 'stimulus_maps')

        # Calculate stimulus maps for each fish in parallel
        timer = Timer()
        timer.start()
        mapper = RandomStimulusMapper(100, experiment, stimulus_map_directory, n_threads=10)
        mapper.run()
        total_time = timer.stop()
        print 'Total time:', timer.convert_time(total_time)
