from datasets.blumenkohl import experiment as blu
from datasets.lakritz import experiment as lak
from datasets.ath5_ablation import experiment as ath5
from datasets.spontaneous import experiment as spontaneous
import pandas as pd
import os


if __name__ == "__main__":

    for experiment in (blu, lak, ath5, spontaneous):
        mapped_bouts = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'),
                                   index_col=0, dtype={'ID': str, 'video_code': str})

        exemplars = pd.read_csv(os.path.join(experiment.parent.subdirs['analysis'], 'exemplars.csv'))
        exemplars = exemplars[exemplars['clean']]
        exemplars = exemplars.reset_index(drop=True)

        mapped_bouts['module'] = exemplars['module'].values[mapped_bouts['exemplar'].values]
        mapped_bouts.to_csv(os.path.join(experiment.subdirs['analysis'], 'mapped_bouts.csv'))
