from datasets.main_dataset import experiment
from paths import stimulus_map_directory
from behaviour_analysis.analysis.stimulus_mapping import find_paramecia
import pandas as pd
import numpy as np
import os


if __name__ == "__main__":

    mapped_bouts = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'),
                               index_col=0, dtype={'ID': str, 'video_code': str})

    jturn_points = []

    for ID, fish_bouts in mapped_bouts.groupby('ID'):
        print ID
        fish_maps = np.load(os.path.join(stimulus_map_directory, ID + '.npy'))
        assert len(fish_maps) == len(fish_bouts)
        module_maps = fish_maps[(fish_bouts['module'] == 6)]
        for frames in module_maps:
            img_points, img_orientations = find_paramecia(frames[0])
            img_points -= np.array([135, 125])
            jturn_points.extend(img_points)

    jturn_points = np.array(jturn_points)

    r = np.linalg.norm(jturn_points, axis=1)
    jturn_points = jturn_points[r <= 80]
    jturn_points = jturn_points[jturn_points[:, 0] > 0]
    # jturn_points[jturn_points[:, 1] < 0] *= (1, -1)

    # theta = np.arctan2(jturn_points[:, 1], jturn_points[:, 0])
    # theta = np.degrees(theta)
    # counts, bins = np.histogram(theta, bins=np.arange(0, 95, 5))

    r = np.linalg.norm(jturn_points, axis=1) * 0.033
    counts, bins = np.histogram(r, bins=np.linspace(0, 1, 20))

    print np.mean(r)

    from matplotlib import pyplot as plt
    plt.plot(np.mean([bins[:-1], bins[1:]], axis=0), counts)
    # plt.scatter(*jturn_points.T, c='k', lw=0, s=1)
    plt.show()
