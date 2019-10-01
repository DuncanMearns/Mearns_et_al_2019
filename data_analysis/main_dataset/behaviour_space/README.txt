Scripts for generating a behavioural space. Before running these scripts, video data should be tracked, kinematics
extracted and bouts identified.

Path to the experiment directory should be set in the setup.py script before running.
Files are saved in {experiment directory}/analysis/behaviour_space with the exception of:
    - exemplars.csv
    - isomap.npy
    - mapped_bouts.csv
which are saved directly in {experiment directory}/analysis.

Scripts should be run in the following order:
    - postural_decomposition.py
    - calculate_distances.py
    - find_exemplars.py
    - check_exemplars.py
    - calculate_exemplar_distances.py
    - isometric_embedding.py
    - find_mapped_bouts.py
    - kinematic_features.py

Postural decomposition
======================
Imports bouts recorded in an experiment and calculates the principal components ("eigenfish") of the tail kinematics.
Generates files:
    - bout_indices.npy
    - eigenfish.npy
    - tail_statistics.npy
    - explained_variance.npy

Calculate distances
===================
Imports bouts recorded in an experiment and maps them onto the principal components. Calculates a distance metric
between each pair of bouts and their mirror images using dynamic time warping.
Generates files:
    - distance_matrix_normal.npy
    - distance_matrix_flipped.npy

Find exemplars
==============
Using the pairwise distances between each pair of bouts, performs affinity propagation to identify a set of exemplars
that are representative of the entire data set.
Generates files:
    - distance_matrix.npy
    - cluster_labels.npy
    - cluster_centres.npy

Check exemplars
===============
Allows the user to manually inspect and verify each exemplar to filter out tracking errors erroneously identified as
bouts.
Generates file:
    - exemplars.csv

Find mapped bouts
=================
Finds the bouts that can be mapped into the behavioural space and assigns each a transition index.
Generates file:
    - mapped_bouts.csv

Calculate exemplar distances
============================
Extracts the distances between verified exemplar bouts from the pairwise distance matrix.
Generates file:
    - exemplar_distance_matrix.npy

Isomap embedding
================
Performs non-linear dimensionality reduction using distances between verified exemplars to represent bouts in a
euclidean space.
Generates files:
    - isomap.npy
    - kernel_pca_eigenvalues.npy
    - reconstruction_errors.npy

Kinematic features
==================
Finds the median of the turn angles, maximum angular velocities, timing of the maximum angular velocity, and mean speeds
across bouts in each cluster.
Generates file:
    - kinematic_features.npy
