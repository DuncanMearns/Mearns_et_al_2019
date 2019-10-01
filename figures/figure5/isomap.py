from plotting import *
from plotting.colors import strike_colors
from datasets.main_dataset import experiment
import pandas as pd
import numpy as np


if __name__ == "__main__":

    capture_strike_directory = os.path.join(experiment.subdirs['analysis'], 'capture_strikes')
    capture_strikes = pd.read_csv(os.path.join(capture_strike_directory, 'capture_strikes.csv'),
                                  index_col=0, dtype={'ID': str, 'video_code': str})
    strike_isomap = np.load(os.path.join(capture_strike_directory, 'isomapped_strikes.npy'))

    # Plot twice (both clustered and unclustered)
    for i in range(2):

        fig, ax = plt.subplots(figsize=(2, 2))
        if i == 0:
            c = '0.5'
        else:
            strike_labels = np.array(['attack', 'sstrike'])[capture_strikes['strike_cluster'].values]
            c = [strike_colors[label] for label in strike_labels]
        ax.scatter(*strike_isomap.T, marker='.', s=10, lw=0, c=c)

        ax.set_xlim(-320, 450)
        ax.set_ylim(-420, 350)
        open_ax(ax)

        ax.spines['left'].set_bounds(-300, 300)
        ax.set_yticks([-300, 0, 300])
        ax.set_yticklabels([-300, 0, 300], fontproperties=verysmallfont)
        ax.set_yticks(np.arange(-300, 300, 100), minor=True)

        ax.spines['left'].set_color('0.5')
        ax.tick_params(axis='y', which='both', color='0.5', labelcolor='0.5')

        ax.spines['bottom'].set_bounds(-300, 400)
        ax.set_xticks([-300, 0, 300])
        ax.set_xticklabels([-300, 0, 300], fontproperties=verysmallfont)
        ax.set_xticks(np.arange(-300, 450, 100), minor=True)

        ax.spines['bottom'].set_color('0.5')
        ax.tick_params(axis='x', which='both', color='0.5', labelcolor='0.5')

        # plt.show()
        if i == 0:
            save_fig(fig, 'figure5', 'strike_isomap')
            plt.close(fig)
        else:
            save_fig(fig, 'figure5', 'strike_isomap_clustered')
