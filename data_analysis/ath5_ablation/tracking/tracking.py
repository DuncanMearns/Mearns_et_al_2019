from datasets.ath5_ablation import experiment


if __name__ == "__main__":

    experiment.update_entries()
    experiment.calculate_backgrounds()
    experiment.set_thresholds(51, track_eyes=False)
    experiment.set_ROIs()

    experiment.run_tracking(51, parallel_processing=True, n_processors=20, track_eyes=False)
    experiment.calculate_kinematics(500., n_processors=20)
    experiment.get_bouts()
