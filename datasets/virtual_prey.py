from behaviour_analysis.manage_files import get_files, get_directories, create_folder
from behaviour_analysis.miscellaneous import print_heading
import pandas as pd
import os

experiment_directory = 'D:\\DATA\\virtual_prey_capture'
data_directory = os.path.join(experiment_directory, 'data')
analysis_directory = create_folder(experiment_directory, 'analysis')

print_heading('Importing data')

fish_data = {}

for date_folder, date_directory in zip(*get_directories(data_directory, return_paths=True)):

    for fish_name, fish_directory in zip(*get_directories(date_directory, return_paths=True)):

        print fish_name

        fish_files, fish_paths = get_files(fish_directory, return_paths=True)

        tracking = pd.read_csv(fish_paths[0], delimiter=';', index_col=0)

        stimulus = pd.read_csv(fish_paths[4], delimiter=';', index_col=0)

        estimator = pd.read_csv(fish_paths[1], delimiter=';', index_col=0)
        threshold = estimator.loc[0, 'threshold']

        metadata = pd.read_json(fish_paths[3])

        fish_data[fish_name] = dict(tracking=tracking,
                                    stimulus=stimulus,
                                    estimator=estimator,
                                    threshold=threshold,
                                    metadata=metadata)

print ''
