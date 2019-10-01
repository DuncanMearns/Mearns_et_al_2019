from datasets.lensectomy import experiment
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

hunting_sequence_directory = os.path.join(experiment.subdirs['analysis'], 'capture_strikes', 'hunting_sequences')

groups = {}
for condition in ('control', 'left', 'right', 'bilateral'):
    groups[condition] = {}
    for strike in ('attack', 'sstrike', 'abort'):
        hist = np.load(os.path.join(hunting_sequence_directory, '{}_{}_histogram.npy'.format(condition, strike)))
        groups[condition][strike] = hist

attack = groups['control']['attack']
attack /= np.apply_over_axes(np.sum, attack, (1, 2))
attack = np.array([gaussian_filter(img, 1.5).T for img in attack])
attack *= 500000
attack[attack > 255] = 255
attack = attack[:, 75:175, 75:175]

sstrike = groups['control']['sstrike']
sstrike /= np.apply_over_axes(np.sum, sstrike, (1, 2))
sstrike = np.array([gaussian_filter(img, 1.5).T for img in sstrike])
sstrike *= 500000
sstrike[sstrike > 255] = 255
sstrike = sstrike[:, 75:175, 75:175]

abort = groups['control']['abort']
abort /= np.apply_over_axes(np.sum, abort, (1, 2))
abort = np.array([gaussian_filter(img, 1.5).T for img in abort])
abort *= 500000
abort[abort > 255] = 255
abort = abort[:, 75:175, 75:175]

import cv2
for g, r, b in zip(sstrike.astype('uint8'), attack.astype('uint8'), abort.astype('uint8')):
    img = np.rollaxis(np.array([b, g, r]), 0, 3)
    cv2.imshow('control', img)
    cv2.waitKey(1)
cv2.destroyAllWindows()
