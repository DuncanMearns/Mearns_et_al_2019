from .main_dataset import experiment as mapping_experiment
from behaviour_analysis.experiments import MappingExperiment

experiment = MappingExperiment('D:\\DATA\\lakritz', mapping_experiment,
                               video_directory='J:\\Duncan Mearns\\prey capture\\lakritz\\videos',
                               conditions=True, log=True)
experiment.open()
