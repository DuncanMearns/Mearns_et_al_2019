from behaviour_analysis.experiments import MappingExperiment
from .main_dataset import experiment as mapping_experiment

experiment = MappingExperiment('F:\\ANALYSIS\\ath5_ablation', mapping_experiment,
                                  video_directory='I:\\Duncan\\Behaviour\\prey_capture_experiments\\ath5_ablation\\videos',
                                  conditions=True, log=True)
experiment.open()
