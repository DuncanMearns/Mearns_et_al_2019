from setup import *
from behaviour_analysis.analysis.embedding import IsomapWeighted
import numpy as np
from scipy.spatial.distance import pdist, squareform


if __name__ == "__main__":

    D = np.load(os.path.join(experiment.subdirs['analysis'], 'behaviour_space', 'exemplar_distance_matrix.npy'))

    USVs = np.load(os.path.join(experiment.subdirs['analysis'], 'transitions', 'USVs.npy'))
    USVa = np.load(os.path.join(experiment.subdirs['analysis'], 'transitions', 'USVa.npy'))

    q, r = np.linalg.qr(np.concatenate([USVs[0, :, 1:2],
                                        USVa[0, :, :2]], axis=1))

    P = squareform(pdist(q))
    mapper = IsomapWeighted(P, n_neighbors=5, n_components=10)
    weighted_isomap = mapper.fit_transform(D)

    np.save(paths['eigenvalues'], mapper.kernel_pca_.lambdas_)
    np.save(paths['weighted_isomap'], weighted_isomap)
