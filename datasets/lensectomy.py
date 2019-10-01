from behaviour_analysis.experiments import MappingExperiment
from .main_dataset import experiment as mapping_experiment

experiment = MappingExperiment('G:\\DATA\\lens_dissection', mapping_experiment, conditions=True, log=True)
experiment.open()
