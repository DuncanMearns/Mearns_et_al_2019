from datasets.lensectomy import experiment
from behaviour_analysis.manage_files import create_folder
from behaviour_analysis.analysis import EyeTrackingData
import os
import numpy as np


if __name__ == "__main__":

    # Paths for saving
    output_directory = create_folder(experiment.subdirs['analysis'], 'eye_convergence')
    convergence_scores_path = os.path.join(output_directory, 'convergence_scores.csv')
    plots_directory = create_folder(output_directory, 'plots')
    distribution_filename = '{}_convergence_distribution.npy'

    # Import data
    eye_tracking = EyeTrackingData.from_experiment(experiment)
    # Calculate thresholds and scores
    convergence_scores = eye_tracking.calculate_convergence_scores(save_plots_to=plots_directory,
                                                                   threshold_limits=(30, 65))
    # Save
    convergence_scores.to_csv(convergence_scores_path, index=False)


    # Combine right and left into unilateral condition
    unilateral_idxs = eye_tracking.metadata[eye_tracking.metadata['condition'].isin(('right', 'left'))]
    eye_tracking.metadata.loc[unilateral_idxs.index, 'condition'] = 'unilateral'
    # Calculate convergence distribution for each condition
    for condition, condition_info in eye_tracking.metadata.groupby('condition'):
        print condition, '\n'
        # Get IDs
        IDs = condition_info['ID'].values
        # Generate mode-adjusted data for each fish in condition
        condition_data = {}
        for ID in IDs:
            fish_data = eye_tracking.data[ID]
            fish_mode = eye_tracking.analysers[ID].mode
            fish_data['convergence'] = fish_data['convergence'] - fish_mode
            condition_data[ID] = fish_data
        # Generate new EyeTrackingData object with mode-adjusted fish
        condition_eye_data = EyeTrackingData(condition_data, metadata=condition_info)
        # Calculate convergence distribution over all fish
        convergence_distribution = condition_eye_data.calculate_convergence_distribution()
        # Save
        output_path = os.path.join(output_directory, distribution_filename.format(condition))
        np.save(output_path, convergence_distribution)
