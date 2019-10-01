from .toy_data import *

state1 = np.array([0, 1, 4])
state2 = np.array([2, 3, 8])
state3 = np.array([5, 6, 7])
states = [state1, state2, state3]
true_labels = [[np.isin(l, s) for s in states].index(True) for l in cluster_labels]

# ----------------------
# Transition structure 1
# ----------------------

P1 = np.zeros((3, 3))
P1[:, 0] = [0.7, 0.2, 0.1]
P1[:, 1] = [0.2, 0.7, 0.1]
P1[:, 2] = [0.1, 0.1, 0.8]

P = np.zeros((n_clusters, n_clusters))
for i in range(3):
    for j in range(3):
        P[np.meshgrid(states[i], states[j])] = P1[i, j] / 3.

transition_structure_1 = dict(P=P, Q=None, states=states, ns=2, na=0)

# ----------------------
# Transition structure 2
# ----------------------

P1 = np.zeros((3, 3))
P1[:, 0] = [0.3, 0.5, 0.2]
P1[:, 1] = [0.1, 0.1, 0.8]
P1[:, 2] = [0.7, 0.1, 0.2]

P = np.zeros((n_clusters, n_clusters))
for i in range(3):
    for j in range(3):
        P[np.meshgrid(states[i], states[j])] = P1[i, j] / 3.

transition_structure_2 = dict(P=P, Q=None, states=states, ns=0, na=1)
