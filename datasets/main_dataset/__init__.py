from behaviour_analysis.experiments import TrackingExperiment2D

video_directory = 'I:\\Duncan\\Behaviour\\prey_capture_experiments\\prey_capture\\videos'
experiment = TrackingExperiment2D('D:\\DATA\\prey_capture', video_directory=video_directory, conditions=False)
experiment.open()
