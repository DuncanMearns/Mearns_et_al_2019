from ...multi_threading import MultiThreading
from .singular_value_decomposition import SVD
import numpy as np


class CrossValidateSVD(MultiThreading):

    def __init__(self, T, n_permutations=1000, n_sym=5, n_asym=5, n_threads=10):
        MultiThreading.__init__(self, n_threads=n_threads)
        self.T = T
        self.indices = np.arange(len(self.T))
        self.n_train = int((len(self.indices) + 1) / 2)
        self.n_permutations = n_permutations
        self.n_sym = n_sym
        self.n_asym = n_asym
        self.sym_result = np.empty((self.n_permutations, self.n_sym + 1))
        self.asym_result = np.empty((self.n_permutations, self.n_asym))

    def _run_on_thread(self, arg):

        # Partition data
        shuffled = np.random.permutation(self.indices)
        train = self.T[shuffled[:self.n_train]].sum(axis=0)
        test = self.T[shuffled[self.n_train:]].sum(axis=0)

        # Normalize test matrix
        test /= test.sum()

        # Compute SVD
        USVs, USVa = SVD(train)
        Us, Ss, Vs = USVs
        Ua, Sa, Va = USVa

        # Compare symmetric transition modes
        symmetric_errors = np.empty((self.n_sym + 1,))
        S = np.array([np.outer(Us[:, i], Vs[:, i]) * Ss[i, i] for i in range(self.n_sym + 1)])
        # Steady state model
        steady_state_model = S[0]
        steady_state_prediction = steady_state_model / steady_state_model.sum()
        symmetric_errors[0] = self.sum_square_errors(steady_state_prediction, test)
        # Add each transition mode to steady state model
        for i, model in enumerate(S[1:]):
            model_prediction = steady_state_model + model
            model_prediction /= model_prediction.sum()
            model_error = self.sum_square_errors(model_prediction, test)
            symmetric_errors[i + 1] = model_error
        self.sym_result[arg] = symmetric_errors

        # Compare antisymmetric transition modes
        antisymmetric_errors = np.empty((self.n_asym))
        A = np.array([np.outer(Ua[:, 2*i], Va[:, 2*i]) * Sa[2*i, 2*i]
                      + np.outer(Ua[:, (2*i)+1], Va[:, (2*i)+1]) * Sa[(2*i)+1, (2*i)+1]
                      for i in range(self.n_asym)])
        # Add each transition mode to steady state model
        for i, model in enumerate(A):
            model_prediction = steady_state_model + model
            model_prediction /= model_prediction.sum()
            model_error = self.sum_square_errors(model_prediction, test)
            antisymmetric_errors[i] = model_error
        self.asym_result[arg] = antisymmetric_errors

    @staticmethod
    def sum_square_errors(A, B):
        return np.square(A - B).sum()

    def run(self):
        self._run(*np.arange(self.n_permutations))
        return self.sym_result, self.asym_result
