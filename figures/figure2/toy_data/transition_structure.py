from plotting import *
import numpy as np

# Sticky
P1 = np.zeros((3, 3))
P1[:, 0] = [0.7, 0.2, 0.1]
P1[:, 1] = [0.2, 0.7, 0.1]
P1[:, 2] = [0.1, 0.1, 0.8]

# Dynamic
P2 = np.zeros((3, 3))
P2[:, 0] = [0.3, 0.5, 0.2]
P2[:, 1] = [0.1, 0.1, 0.8]
P2[:, 2] = [0.7, 0.1, 0.2]

fig, axes = plt.subplots(1, 2, figsize=(2, 1))
axes[0].matshow(P1, cmap='plasma', vmin=0, vmax=1.0)
axes[1].matshow(P2, cmap='plasma', vmin=0, vmax=1.0)
save_fig(fig, 'figureS2', 'transition_structure_matrix')
