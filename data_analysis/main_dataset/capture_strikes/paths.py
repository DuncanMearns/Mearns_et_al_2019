from datasets.main_dataset import experiment
from behaviour_analysis.manage_files import create_folder
import os

capture_strike_directory = create_folder(experiment.subdirs['analysis'], 'capture_strikes')

paths = {}
for fname in ('capture_strikes', 'strike_frames'):
    paths[fname] = os.path.join(capture_strike_directory, fname + '.csv')
for fname in ('capture_strike_distance_matrix', 'isomapped_strikes'):
    paths[fname] = os.path.join(capture_strike_directory, fname + '.npy')
