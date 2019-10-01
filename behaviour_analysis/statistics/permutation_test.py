from ..multi_threading import MultiThreading
import numpy as np


class PermutationTest(MultiThreading):

    def __init__(self, values, labels, func, n_permutations, **kwargs):
        MultiThreading.__init__(self, **kwargs)
        assert len(values) == len(labels)
        self.values = values
        self.labels = labels
        self.func = func
        self.n_permutations = n_permutations

    def _run_on_thread(self, arg):
        labels = np.random.permutation(self.labels)
        u_values = self.values[labels == 0]
        v_values = self.values[labels == 1]
        stat = self.func(u_values, v_values)
        self.null_distribution[arg] = stat

    def _calculate_null_distribution(self):
        self.null_distribution = np.empty((self.n_permutations,))
        self._run(*np.arange(self.n_permutations))

    def _calculate_test_statistic(self):
        u_values = self.values[self.labels == 0]
        v_values = self.values[self.labels == 1]
        stat = self.func(u_values, v_values)
        self.test_statistic = stat

    def run(self):
        self._calculate_null_distribution()
        self._calculate_test_statistic()
        return self.test_statistic, self.null_distribution
