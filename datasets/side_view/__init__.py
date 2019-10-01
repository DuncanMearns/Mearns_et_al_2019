from behaviour_analysis.experiments import TrackingExperiment3D
from behaviour_analysis.manage_files import create_folder

experiment = TrackingExperiment3D('D:\\DATA\\3D_prey_capture')
experiment.open()
experiment.subdirs['analysis'] = create_folder(experiment.directory, 'analysis')
