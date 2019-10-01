from datasets.lensectomy import experiment
from paths import stimulus_map_directory
from behaviour_analysis.analysis.stimulus_mapping import RandomStimulusMapper
from behaviour_analysis.miscellaneous import Timer


if __name__ == "__main__":

    # Calculate stimulus maps for each fish in parallel
    timer = Timer()
    timer.start()
    mapper = RandomStimulusMapper(100, experiment, stimulus_map_directory, n_threads=10)
    mapper.run()
    total_time = timer.stop()
    print 'Total time:', timer.convert_time(total_time)
