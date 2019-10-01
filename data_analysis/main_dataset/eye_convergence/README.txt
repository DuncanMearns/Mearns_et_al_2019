Scripts for analysing eye angle data.

Paths to the experiment directory should be set in the setup.py script before running. Outputs files are saved in:
{experiment_directory}\\analysis\\eye_convergence

Scripts should be run in the following order:
    - eye_convergence_analysis.py
    - behaviour_phase_classification.py
    - state_eye_convergence.py

Eye convergence analysis
========================
Imports eye angle data from all animals in an experiment, calculates their individual convergence scores and thresholds,
and computes 1D and 2D kernel density estimates of the convergence and left-right eye angle distributions over all
animals.
Generates files:
    - convergence_scores.csv
    - eye_convergence_counts.npy
    - eye_angle_counts.npy
    - {ID}.png for each animal in the experiment in the folder "plots"

Behaviour phase classification
==============================
Imports all bouts that are mapped to a behavioural space, computes the convergence angle of the eyes before and after
each bout, and assigns each a convergence state.
Generates file:
    - convergence_states.npy
    - state_convergence_scores.npy
