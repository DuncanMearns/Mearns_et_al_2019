from datasets.lensectomy import experiment


if __name__ == "__main__":
    experiment.update_entries()
    experiment.calculate_backgrounds()
    experiment.set_thresholds(n_points=51)
    experiment.set_ROIs()
    experiment.run_tracking(n_points=51)
    experiment.calculate_kinematics(frame_rate=500.)
    experiment.get_bouts()
