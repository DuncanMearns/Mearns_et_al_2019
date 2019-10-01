from .main_dataset import experiment as mapping_experiment
from behaviour_analysis.experiments import MappingExperiment

experiment = MappingExperiment('D:\\DATA\\blu_s257', mapping_experiment,
                               video_directory='J:\\Duncan Mearns\\prey capture\\blu_s257\\videos',
                               conditions=True, log=True)
experiment.open()
