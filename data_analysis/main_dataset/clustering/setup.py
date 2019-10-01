from behaviour_analysis.experiments import TrackingExperiment2D
from behaviour_analysis.manage_files import create_folder
import os


video_directory = 'I:\\Duncan\\Behaviour\\prey_capture_experiments\\prey_capture\\videos'
experiment = TrackingExperiment2D('D:\\DATA\\prey_capture', video_directory=video_directory, log=False)
experiment.open()
experiment.subdirs['analysis'] = create_folder(experiment.directory, 'analysis')

clustering_directory = create_folder(experiment.subdirs['analysis'], 'clustering')

paths = {}
paths['exemplars'] = os.path.join(experiment.subdirs['analysis'], 'exemplars.csv')
paths['mapped_bouts'] = os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv')
for fname in ['eigenvalues', 'weighted_isomap']:
    paths[fname] = os.path.join(clustering_directory, fname + '.npy')
