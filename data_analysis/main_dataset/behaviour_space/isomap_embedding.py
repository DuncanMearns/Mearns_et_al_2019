from paths import paths
from behaviour_analysis.analysis.embedding import IsomapPrecomputed
import numpy as np


if __name__ == "__main__":

    exemplar_distance_matrix = np.load(paths['exemplar_distance_matrix'])

    isomap = IsomapPrecomputed(n_components=20, n_neighbors=5)

    isomap_embedding = isomap.fit_transform(exemplar_distance_matrix)
    for i in (0, 2):  # mirror axes 0 and 2
        isomap_embedding[:, i] *= -1
    kernel_pca_eigenvalues = isomap.kernel_pca_.lambdas_
    reconstruction_errors = isomap.reconstruction_errors()

    np.save(paths['isomap'], isomap_embedding)
    np.save(paths['kernel_pca_eigenvalues'], kernel_pca_eigenvalues)
    np.save(paths['reconstruction_errors'], reconstruction_errors)
