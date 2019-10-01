from datasets.blumenkohl import experiment as blu
from datasets.lakritz import experiment as lak
from datasets.main_dataset import experiment as mapping_experiment
from behaviour_analysis.analysis.bouts import BoutData, import_bouts
from behaviour_analysis.analysis.alignment.distance import calculate_distance_matrix_templates
from behaviour_analysis.manage_files import create_folder, create_filepath
from behaviour_analysis.miscellaneous import print_heading, Timer
import numpy as np
import pandas as pd
import os


if __name__ == "__main__":

    #################
    n_dims = 3
    frame_rate = 500.
    #################

    # Import data from mapping experiment
    mapping_space_directory = os.path.join(mapping_experiment.subdirs['analysis'], 'behaviour_space')
    eigenfish = np.load(os.path.join(mapping_space_directory, 'eigenfish.npy'))
    mean_tail, std_tail = np.load(os.path.join(mapping_space_directory, 'tail_statistics.npy'))
    exemplar_info = pd.read_csv(os.path.join(mapping_experiment.subdirs['analysis'], 'exemplars.csv'),
                                index_col='bout_index',
                                dtype={'ID': str, 'video_code': str})
    exemplar_info = exemplar_info[exemplar_info['clean']]
    exemplars = BoutData.from_directory(exemplar_info, mapping_experiment.subdirs['kinematics'],
                                        check_tail_lengths=False, tail_columns_only=True)
    exemplars = exemplars.map(eigenfish, whiten=True, mean=mean_tail, std=std_tail)
    exemplars = exemplars.list_bouts(values=True, ndims=n_dims)

    for experiment in (blu, lak):

        # Set paths
        output_directory = create_folder(experiment.subdirs['analysis'], 'distance_matrices')

        # Import experiment bouts
        experiment_bouts = import_bouts(experiment.directory)
        experiment_bouts = experiment_bouts.map(eigenfish, whiten=True, mean=mean_tail, std=std_tail)

        # Compute distance matrices
        print_heading('CALCULATING DISTANCE MATRICES')
        distances = {}
        analysis_times = []
        timer = Timer()
        timer.start()
        for ID in experiment_bouts.metadata['ID'].unique():
            output_path, path_exists = create_filepath(output_directory, ID, '.npy', True)
            if path_exists:
                distances[ID] = np.load(output_path)
            if not path_exists:
                print ID + '...',
                queries = experiment_bouts.list_bouts(IDs=[ID], values=True, ndims=n_dims)
                fish_distances = calculate_distance_matrix_templates(queries, exemplars, fs=frame_rate)
                distances[ID] = fish_distances
                time_taken = timer.lap()
                analysis_times.append(time_taken)
                print timer.convert_time(time_taken)
                np.save(output_path, fish_distances)
        if len(analysis_times) > 0:
            print 'Average time: {}'.format(timer.convert_time(timer.average))

        # Assign exemplars
        mapped_bouts = experiment_bouts.metadata.copy()
        mapped_bouts['exemplar'] = None
        for ID, fish_distances in distances.iteritems():
            bout_idxs = mapped_bouts[mapped_bouts['ID'] == ID].index
            nearest_exemplar = np.argmin(fish_distances, axis=1)
            mapped_bouts.loc[bout_idxs, 'exemplar'] = nearest_exemplar
        mapped_bouts.to_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'))
