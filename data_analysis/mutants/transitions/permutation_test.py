from datasets.blumenkohl import experiment as blu
from datasets.lakritz import experiment as lak
from behaviour_analysis.analysis.transitions import SVD
from behaviour_analysis.miscellaneous import Timer, print_heading
import itertools
import numpy as np
import os


def permute(idxs, n, exact=False, n_permutations=1000):
    if exact:
        for group1 in itertools.combinations(idxs, n):
            group1 = np.array(group1)
            group2 = idxs[~np.isin(idxs, group1)]
            yield group1, group2
    else:
        for i in range(n_permutations):
            random_idxs = np.random.permutation(idxs)
            group1 = random_idxs[:n]
            group2 = random_idxs[n:]
            yield group1, group2


def compare_transition_modes(m1, m2, exact=False, n_permutations=1000):
    """Compare the transitions matrices between two groups of animals.

    Parameters
    ----------
    m1, m2: np.ndarray, shape (n_animals, n_states, n_states)
        Transition matrices for the two groups of animals
    exact : bool, default False
        Whether to compute every possible permutation
    n_permutations
        Number of permutations to compute if exact is False

    Returns
    -------
    dot_products : np.ndarray, shape (N, 3)
        N x 3 matrix containing dot products between first two symmetric and first antisymmetric transition modes across
        N permutations of animals
    """
    # Number of animals in test group
    n_test = len(m1)
    # Concatenate transition matrices
    all_transitions = np.concatenate([m1, m2], axis=0)
    idxs = np.arange(len(all_transitions))
    # Output
    dot_products = []
    # Iterate through different permutations of animals
    i = 0
    for group1, group2 in permute(idxs, n_test, exact=exact, n_permutations=n_permutations):
        if i % 10 == 0:
            print i
        # Sum transition matrices
        T1 = all_transitions[group1].sum(axis=0)
        T2 = all_transitions[group2].sum(axis=0)
        # Compute the SVD
        USVs1, USVa1 = SVD(T1)
        USVs2, USVa2 = SVD(T2)
        # Compare first two symmetric transition modes
        s1_dot = np.abs(np.dot(USVs1[0, :, 0], USVs2[0, :, 0]))
        s2_dot = np.abs(np.dot(USVs1[0, :, 1], USVs2[0, :, 1]))
        # Compare first antisymmetric transition mode
        a1_dot_a = np.abs(np.dot(USVa1[0, :, 0], USVa2[0, :, 0]))
        a1_dot_b = np.abs(np.dot(USVa1[0, :, 0], USVa2[0, :, 1]))
        a1_dot = max(a1_dot_a, a1_dot_b)
        # Append to output
        dot_products.append((s1_dot, s2_dot, a1_dot))
        i += 1
    return np.array(dot_products)


if __name__ == "__main__":

    # ==========
    # Blumenkohl
    # ==========

    # print_heading('Blumenkohl')
    #
    # transition_directory = os.path.join(blu.subdirs['analysis'], 'transitions')
    # control_matrices = np.load(os.path.join(transition_directory, 'het', 'transition_matrices.npy'))
    # mutant_matrices = np.load(os.path.join(transition_directory, 'mut', 'transition_matrices.npy'))
    #
    # timer = Timer()
    # timer.start()
    # dot_products = compare_transition_modes(mutant_matrices, control_matrices, exact=False, n_permutations=1000)
    # np.save(os.path.join(transition_directory, 'compare_control_mutant.npy'), dot_products)
    # print timer.convert_time(timer.stop())
    # print '\n'

    # =======
    # Lakritz
    # =======

    print_heading('Lakritz')

    transition_directory = os.path.join(lak.subdirs['analysis'], 'transitions')
    control_matrices = np.load(os.path.join(transition_directory, 'ctrl', 'transition_matrices.npy'))
    mutant_matrices = np.load(os.path.join(transition_directory, 'mut', 'transition_matrices.npy'))

    timer = Timer()
    timer.start()
    dot_products = compare_transition_modes(mutant_matrices, control_matrices, exact=True)
    np.save(os.path.join(transition_directory, 'compare_control_mutant.npy'), dot_products)
    print timer.convert_time(timer.stop())
