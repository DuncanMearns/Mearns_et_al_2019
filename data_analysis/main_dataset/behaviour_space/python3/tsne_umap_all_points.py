import numpy as np
from umap import UMAP
from sklearn.manifold import TSNE
import time
from scipy.spatial.distance import squareform


if __name__ == "__main__":

    distance_matrix_path = 'D:\\Duncan\\Data\\distance_matrix.npy'
    # umap_path = 'D:\\Duncan\\Data\\umap_embedding_{}.npy'
    # tsne_path = 'D:\\Duncan\\Data\\tsne_embedding_{}.npy'
    umap_path = 'D:\\Duncan\\Data\\umap_embedding_{}_1511.npy'
    tsne_path = 'D:\\Duncan\\Data\\tsne_embedding_{}_1511.npy'

    np.random.seed(1511)

    print('Importing distance matrix...', end=' ')
    D = np.load(distance_matrix_path)
    print(D.min(), D.max())
    D = squareform(D)
    print('done!')

    for n_neighbors in [20]:
    # for n_neighbors in [5, 10, 20, 50]:

        print('Embedding with {} nearest neighbors'.format(n_neighbors))

        print('\tStarting UMAP')
        umap = UMAP(n_components=2, n_neighbors=n_neighbors, metric='precomputed')
        start = time.time()
        embedding = umap.fit_transform(D)
        end = time.time()
        np.save(umap_path.format(n_neighbors), embedding)
        print('\tUMAP time: {} minutes\n'.format((end - start) / 60.))

        print('\tStarting t-SNE')
        tsne = TSNE(n_components=2, perplexity=n_neighbors, metric='precomputed')
        start = time.time()
        embedding = tsne.fit_transform(D)
        end = time.time()
        np.save(tsne_path.format(n_neighbors), embedding)
        print('\tTSNE time: {} minutes\n'.format((end - start) / 60.))
