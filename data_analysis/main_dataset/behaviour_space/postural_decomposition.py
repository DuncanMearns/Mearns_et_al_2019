from datasets.main_dataset import experiment
from paths import paths
from behaviour_analysis.analysis.bouts import BoutData
import numpy as np


if __name__ == "__main__":

    bouts = BoutData.from_directory(paths['bouts'], experiment.subdirs['kinematics'],
                                    tail_columns_only=True, check_tail_lengths=True)
    bout_indices = bouts.metadata.index
    np.save(paths['bout_indices'], bout_indices)

    transformed, pca = bouts.transform()

    eigenfish = pca.components_
    tail_statistics = np.array([bouts.mean.values, bouts.std.values])
    explained_variance = pca.explained_variance_ratio_

    np.save(paths['eigenfish'], eigenfish)
    np.save(paths['tail_statistics'], tail_statistics)
    np.save(paths['explained_variance'], explained_variance)
