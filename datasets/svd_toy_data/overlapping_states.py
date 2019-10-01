from .toy_data import *

state1 = np.array([0, 1, 8, 4])
state2 = np.array([2, 3, 7, 8])
state3 = np.array([1, 5, 6, 7])
states = [state1, state2, state3]

# True labels
idxs = np.arange(9)
in1, in2, in3 = np.isin(idxs, state1), np.isin(idxs, state2), np.isin(idxs, state3)
clusters = np.array([idxs[(in1) & (~in2) & (~in3)],  # state 1 only
                     idxs[(~in1) & (in2) & (~in3)],  # state 2 only
                     idxs[(~in1) & (~in2) & (in3)],  # state 3 only
                     idxs[(in1) & (in2) & (~in3)],   # states 1 and 2
                     idxs[(in1) & (~in2) & (in3)],   # states 1 and 3
                     idxs[(~in1) & (in2) & (in3)]])  # states 2 and 3
true_labels = [[np.isin(l, cluster) for cluster in clusters].index(True) for l in cluster_labels]

# ----------------------
# Transition structure 3
# ----------------------

Q = np.zeros((3, 3))
Q[:, 0] = [0.7, 0.2, 0.1]
Q[:, 1] = [0.2, 0.7, 0.1]
Q[:, 2] = [0.1, 0.1, 0.8]

P = np.zeros((3, n_clusters, n_clusters))
for i, state in enumerate(states):
    P[i][np.meshgrid(state, state)] = np.ones((len(state), len(state)))* 0.25

transition_structure_3 = dict(P=P, Q=Q, states=states, ns=2, na=0)

# ----------------------
# Transition structure 4
# ----------------------

Q = np.zeros((3, 3))
Q[:, 0] = [0.3, 0.5, 0.2]
Q[:, 1] = [0.1, 0.1, 0.8]
Q[:, 2] = [0.7, 0.1, 0.2]

P = np.zeros((3, n_clusters, n_clusters))
for i, state in enumerate(states):
    P[i][np.meshgrid(state, state)] = np.ones((len(state), len(state)))* 0.25

transition_structure_4 = dict(P=P, Q=Q, states=states, ns=0, na=1)
