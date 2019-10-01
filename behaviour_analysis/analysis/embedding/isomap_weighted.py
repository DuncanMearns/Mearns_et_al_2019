from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils.graph import graph_shortest_path


class IsomapWeighted(Isomap):

    def __init__(self, W, n_neighbors=5, n_components=2,
                 eigen_solver='auto', tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto', n_jobs=-1):

        Isomap.__init__(self, n_neighbors=n_neighbors, n_components=n_components,
                        eigen_solver=eigen_solver, tol=tol, max_iter=max_iter, path_method=path_method,
                        neighbors_algorithm=neighbors_algorithm, n_jobs=n_jobs)
        self.W = W

    def _fit_transform(self, X):
        assert X.shape[0] == X.shape[1]
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      algorithm=self.neighbors_algorithm,
                                      metric='precomputed',
                                      n_jobs=self.n_jobs)
        self.nbrs_.fit(X)
        self.training_data_ = self.nbrs_._fit_X
        self.kernel_pca_ = KernelPCA(n_components=self.n_components,
                                     kernel="precomputed",
                                     eigen_solver=self.eigen_solver,
                                     tol=self.tol, max_iter=self.max_iter,
                                     n_jobs=self.n_jobs)

        kng = kneighbors_graph(self.nbrs_, self.n_neighbors, metric='precomputed',
                               mode='distance', n_jobs=self.n_jobs)
        kng_weighted = kng.toarray() * self.W

        self.dist_matrix_ = graph_shortest_path(kng_weighted,
                                                method=self.path_method,
                                                directed=False)
        G = self.dist_matrix_ ** 2
        G *= -0.5

        self.embedding_ = self.kernel_pca_.fit_transform(G)
