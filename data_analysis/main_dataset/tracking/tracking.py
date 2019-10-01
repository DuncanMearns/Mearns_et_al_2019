from datasets.main_dataset import experiment


if __name__ == "__main__":

    # Update fish entries
    experiment.update_entries()

    # Perform tracking
    experiment.calculate_backgrounds()
    experiment.set_thresholds(n_points=51)
    experiment.set_ROIs()
    experiment.run_tracking(n_points=51)
    experiment.calculate_kinematics(frame_rate=500.)
    experiment.get_bouts()
