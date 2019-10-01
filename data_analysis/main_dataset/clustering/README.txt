Scripts for finding behavioural modules based on similarity of kinematics and transitions.

Path to the experiment directory should be set in the setup.py script before running.
Files are saved in {experiment directory}/analysis/clustering.

Scripts should be run in the following order:
    - kinematic_transition_space.py
    - clustering.py

Kinematic transition space
==========================
Performs weighted isomap embedding using dtw distances between exemplars (kinematic similarity) and distances between
exemplars in the transition space defined by the singular vectors.
Generates files:
    - weighted_isomap.npy
    - eigenvalues.npy

Clustering
==========
Performs k-means clustering in the kinematic-transitions space to find behavioural modules.
Adds a 'module' column to the exemplar and mapped bouts DataFrames.
