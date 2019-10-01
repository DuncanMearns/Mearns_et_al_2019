from behaviour_analysis.analysis.transitions import generate_weights
from scipy.stats import norm
from scipy.cluster import hierarchy as sch
import numpy as np


np.random.seed(1992)
n_clusters = 9

cluster_centres = np.random.rand(n_clusters, 2)
states_per_cluster = norm.rvs(loc=50, scale=5, size=n_clusters).astype('i4')

points = []
cluster_labels = []
l = 0
for centre, n_states in zip(cluster_centres, states_per_cluster):
    xs = norm.rvs(loc=centre[0], scale=0.06, size=n_states)
    ys = norm.rvs(loc=centre[1], scale=0.06, size=n_states)
    ps = np.vstack([xs, ys]).T
    points.append(ps)
    cluster_labels.append(np.ones((len(ps),)) * l)
    l += 1
points = np.concatenate(points)
cluster_labels = np.concatenate(cluster_labels)

n_points = len(points)
random_point_order = np.random.permutation(np.arange(n_points))

points = points[random_point_order]
cluster_labels = cluster_labels[random_point_order]
n_points = n_points


class ToyData():

    def __init__(self, P, Q=None, **kwargs):
        if Q is not None:
            assert len(Q) == len(P)
        self.P = P
        self.Q = Q

    @property
    def n_states(self):
        return self.P.shape[-1]

    @property
    def n_hidden_states(self):
        if self.Q is None:
            return 0
        else:
            return len(self.Q)

    def _markov_process(self, n_transitions):

        current_state = np.random.choice(np.arange(self.n_states))
        current_point = np.random.choice(np.arange(n_points)[cluster_labels == current_state])

        T = np.zeros((n_points, n_points))

        for t in range(n_transitions):
            new_state = np.random.choice(np.arange(self.n_states), p=self.P[:, current_state])
            new_point = np.random.choice(np.arange(n_points)[cluster_labels == new_state])
            T[new_point, current_point] += 1
            current_state = new_state
            current_point = new_point

        return T

    def _hidden_markov_process(self, n_transitions):

        current_hidden_state = np.random.choice(np.arange(self.n_hidden_states))
        P = self.P[current_hidden_state]
        current_state = np.random.choice(np.where(P > 0)[0])
        current_point = np.random.choice(np.arange(n_points)[cluster_labels == current_state])

        T = np.zeros((n_points, n_points))

        for t in range(n_transitions):
            new_hidden_state = np.random.choice(np.arange(self.n_hidden_states), p=self.Q[:, current_hidden_state])
            if new_hidden_state == current_hidden_state:
                new_state = np.random.choice(np.arange(self.n_states), p=P[:, current_state])
            else:
                P = self.P[new_hidden_state]
                new_state = np.random.choice(np.where(P > 0)[0])
            new_point = np.random.choice(np.arange(n_points)[cluster_labels == new_state])
            T[new_point, current_point] += 1
            current_hidden_state = new_hidden_state
            current_state = new_state
            current_point = new_point

        return T

    def markov_process(self, n_transitions=10000):
        if self.Q is None:
            T = self._markov_process(n_transitions)
        else:
            T = self._hidden_markov_process(n_transitions)
        return T


class TransitionStructureAnalysis():

    def __init__(self, points, T):
        self.points = points
        self.T = T
        self.T_ = T

    def redistribute_transitions(self, bandwidth):
        self.W = generate_weights(self.points, bandwidth=bandwidth)
        self.T_ = np.dot(np.dot(self.W, self.T), self.W.T)

    def svd(self):
        Us, Ss, VsT = np.linalg.svd(0.5 * (self.T_ + self.T_.T))
        Ua, Sa, VaT = np.linalg.svd(0.5 * (self.T_ - self.T_.T))
        self.Us = Us
        self.Ss = Ss
        self.Vs = VsT.T
        self.Ua = Ua
        self.Sa = Sa
        self.Va = VaT.T

    def transition_space(self, ns, na):
        q, r = np.linalg.qr(np.concatenate([self.Vs[:, 1:1 + ns],
                                            self.Va[:, :2 * na]], axis=1))
        self.q = q

    def dendrogram(self):
        self.Y = sch.linkage(self.q, method='ward')
