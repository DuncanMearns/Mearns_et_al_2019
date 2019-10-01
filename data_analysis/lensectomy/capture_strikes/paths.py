from datasets.lensectomy import experiment
from behaviour_analysis.manage_files import create_folder
import os

capture_strike_directory = create_folder(experiment.subdirs['analysis'], 'capture_strikes')

paths = {}
paths['capture_strikes'] = os.path.join(capture_strike_directory, 'capture_strikes.csv')
for fname in ('distance_matrix', 'isomapped_strikes'):
    paths[fname] = os.path.join(capture_strike_directory, fname + '.npy')
