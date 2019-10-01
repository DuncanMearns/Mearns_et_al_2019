from datasets.main_dataset import experiment
from behaviour_analysis.manage_files import create_folder
import os
import numpy as np
from behaviour_analysis.analysis.eye_convergence import EyeTrackingData


eye_convergence_directory = create_folder(experiment.subdirs['analysis'], 'eye_convergence')
convergence_scores_path = os.path.join(eye_convergence_directory, 'convergence_scores.csv')


if __name__ == "__main__":

    eye_tracking = EyeTrackingData.from_experiment(experiment)

    plots_directory = create_folder(eye_convergence_directory, 'plots')
    convergence_scores = eye_tracking.calculate_convergence_scores(save_plots_to=plots_directory)
    convergence_scores.to_csv(convergence_scores_path, index=False)

    convergence_counts = eye_tracking.calculate_convergence_distribution()
    np.save(os.path.join(eye_convergence_directory, 'eye_convergence_counts.npy'), convergence_counts)

    angle_counts = eye_tracking.calculate_angle_distribution()
    np.save(os.path.join(eye_convergence_directory, 'eye_angle_counts.npy'), angle_counts)
