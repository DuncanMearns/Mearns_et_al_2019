Scripts for analysing capture strikes.

Path to the experiment directory should be set in the setup.py script before running.
Files are saved in {experiment directory}/analysis/capture_strikes.

Scripts should be run in the following order:
    - find_capture_strikes.py
    - calculate_strike_distances.py
    - capture_strike_isomap.py
    - classify_strikes.py
    - check_strikes.py
    - compute_stimulus_maps.py
    - capture_strike_z_score_maps.py
    - annotate_captures.py

Find capture strikes
======================
Finds capture strikes in the experiment.
Generates file:
    - capture_strikes.csv

Calculate strike distances
==========================
Computes the dynamic time warping distances between the first 50ms of each pair capture strikes.
Generates file:
    - capture_strike_distance_matrix.npy

Capture strike isomap
=====================
Generate a capture strike subspace using the capture strike distance matrix.
Generates file:
    - isomapped_strikes.npy

Classify strikes
================
Perform KMeans clustering in the capture strike subspace.
Adds a 'strike_cluster' column to the capture_strikes.csv file.

Check strikes
=============
Plays randomly selected capture strikes from each cluster. Visualisation only.

Compute stimulus sequences
==========================
Generates a background-subtracted and head-centred sequence of frames for each hunting event.
Generates files in hunting_sequences subdirectory:
    - hunting_events.csv
    - {ID}.npy for each fish ID

Compute density maps
====================
Calculates a 2D histograms of paramecia counts across all hunting sequences resulting in either an attack swim or
s-strike.
Generates files in hunting_sequences subdirectory:
    - attack_histogram.npy
    - sstrike_histogram.npy

Annotate captures
=================
Selects 100 random capture strikes and allows user to annotate frames at which the jaw is maximally extended.
Generates file:
    - strike_frames.csv
