from behaviour_analysis.experiments import MappingExperiment
from .main_dataset import experiment as mapping_experiment

experiment = MappingExperiment('D:\\DATA\\spontaneous', mapping_experiment,
                               video_directory='I:\\Duncan\\Behaviour\\prey_capture_experiments\\spontaneous\\videos',
                               conditions=False, log=True)
experiment.open()
