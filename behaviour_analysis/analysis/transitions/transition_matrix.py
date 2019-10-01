import numpy as np


def transition_matrix(seq, index=None, n_states=None):
    """Compute a transition matrix from a sequence of states.

    Parameters
    ----------
    seq : np.array
        Sequence of states
    index : np.array
        Sequence of indices for given states
    n_states : int
        Total number of states

    Returns
    -------
    T : np.array, shape: (n_states, n_states)
        Number of transitions between each pair of states. T[i, j] = n_transitions(j -> i)
    """
    if n_states is None:
        n_states = seq.max() + 1
    if index is None:
        index = np.arange(len(seq))
    T = np.zeros((n_states, n_states))
    t0 = np.where(np.diff(index) == 1)[0]
    transitions = np.array([seq[t0], seq[t0 + 1]]).T
    for s0, s1 in transitions:
        T[s1, s0] += 1

    return T
