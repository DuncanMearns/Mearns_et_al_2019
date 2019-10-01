from ...miscellaneous import print_heading, read_csv
from ...manage_files import create_folder

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def import_bouts(experiment_directory):
    analysis_directory = create_folder(experiment_directory, 'analysis')
    bouts_path = os.path.join(experiment_directory, 'bouts.csv')
    kinematics_directory = os.path.join(experiment_directory, 'kinematics')
    bout_indices_path = os.path.join(analysis_directory, 'bout_indices.npy')

    if os.path.exists(bout_indices_path):
        bout_indices = np.load(bout_indices_path)
        bouts_df = pd.read_csv(bouts_path, dtype={'ID': str, 'video_code': str})
        bouts_df = bouts_df.loc[bout_indices]
        bouts = BoutData.from_directory(bouts_df, kinematics_directory,
                                        tail_columns_only=True, check_tail_lengths=False)
    else:
        bouts = BoutData.from_directory(bouts_path, kinematics_directory, tail_columns_only=True)
        bout_indices = bouts.metadata.index
        np.save(bout_indices_path, bout_indices)
    return bouts


def open_kinematics(bouts_df, kinematics_directory):
    # =========================
    # IMPORT ALL KINEMATIC DATA
    # =========================
    print_heading('IMPORTING DATA')
    # Initialise lists for storing DataFrames and indices
    kinematics_dfs = []
    multi_index_keys = []
    # Iterate through fish IDs
    for ID, fish_bouts in bouts_df.groupby('ID'):
        print ID
        print '-' * len(ID)
        # Iterate through video codes
        for video_code, video_bouts in fish_bouts.groupby('video_code'):
            print video_code
            # Open kinematics DataFrame
            kinematics_path = os.path.join(kinematics_directory, ID, video_code + '.csv')
            df = pd.read_csv(kinematics_path)
            # Assign frames to bouts
            bout_indexer = np.zeros((len(df),), dtype='i4') - 1
            for idx, bout_info in video_bouts.iterrows():
                bout_indexer[bout_info.start:bout_info.end] = idx
            bout_indexer = zip(bout_indexer, df.index)
            df.index = pd.MultiIndex.from_tuples(bout_indexer, names=['bout_index', 'frame'])
            # Append DataFrame and indices to lists
            kinematics_dfs.append(df)
            multi_index_keys.append((ID, video_code))
        print ''
    # Create the multi-indexed data attribute
    data = pd.concat(kinematics_dfs, keys=multi_index_keys, names=['ID', 'video_code'])
    return data


def check_tail_lengths_for_bouts(data):
    print 'Checking tail lengths...',
    # Initialise list for checking the tail length
    bout_tail_length_checks = []
    # Iterate through fish IDs
    for ID, fish_data in data.groupby('ID'):
        # Get all bout frames
        fish_bout_frames = fish_data[fish_data.index.get_level_values('bout_index') >= 0]
        fish_bout_tail_lengths = fish_bout_frames['length']
        # Calculate the minimum tail length that is not considered a tracking error (half of the mode length)
        max_length = np.ceil(fish_bout_tail_lengths.max())
        counts, bins = np.histogram(fish_bout_tail_lengths, bins=np.arange(max_length))
        mode_length = counts.argmax()
        min_length = mode_length / 2.
        # Iterate through bouts to check whether a tracking error occurred
        tail_lengths_check = fish_data['length'] > min_length
        bout_tail_length_checks += [np.all(bout) for bout_index, bout in tail_lengths_check.groupby('bout_index') if bout_index >= 0]
    print 'done!'
    return bout_tail_length_checks


def whiten_data(data, mean=None, std=None):
    if (mean is None) and (std is None):
        whitened = (data - data.mean(axis=0)) / data.std(axis=0)
    elif (mean is not None) and (std is not None):
        whitened = (data - mean) / std
    else:
        raise ValueError('both mean and std must be specified!')
    return whitened


def transform_data(data):
    pca = PCA()
    transformed = pca.fit_transform(data)
    return transformed, pca


def map_data(data, vectors):
    assert vectors.shape[1] == data.shape[1]
    mapped = np.dot(data, vectors.T)
    return mapped


class BoutData(object):

    def __init__(self, data, metadata=None):
        self.data = data
        assert isinstance(self.data.index, pd.core.index.MultiIndex), 'Data must be MultiIndexed!'
        self.metadata = metadata

    @property
    def mean(self):
        return self.data.mean(axis=0)

    @property
    def std(self):
        return self.data.std(axis=0)

    @classmethod
    def from_directory(cls, bouts, kinematics_directory, check_tail_lengths=True, tail_columns_only=True):
        # Open the bouts DataFrame
        if type(bouts) == pd.DataFrame:
            bouts_df = bouts
        elif type(bouts) == str:
            bouts_df = read_csv(bouts, ID=str, video_code=str)
        else:
            raise TypeError('bouts must be string or DataFrame')
        # Import kinematics data
        data = open_kinematics(bouts_df, kinematics_directory)
        if check_tail_lengths:
            keep_bouts = check_tail_lengths_for_bouts(data)
            bouts_df = bouts_df[keep_bouts]
        try:
            kept_bout_data = bouts_df[bouts_df['ROI']]
        except KeyError:  # ROI is not in bouts_df DataFrame, assume all bouts lie within ROI
            kept_bout_data = bouts_df
        data = data[data.index.get_level_values('bout_index').isin(kept_bout_data.index)]
        if tail_columns_only:
            tail_columns = [col for col in data.columns if col[0] == 'k']
            data = data[tail_columns]
        return cls(data, kept_bout_data)

    def _slice_data(self, **kwargs):
        keys = kwargs.keys()
        # Create indexer for IDs and video codes
        indexer = dict(IDs=slice(None), video_codes=slice(None))
        for key in ['IDs', 'video_codes']:
            if key in keys and kwargs[key]:
                    indexer[key] = kwargs[key]
            elif key[:-1] in keys and kwargs[key[:-1]]:
                    indexer[key] = (kwargs[key[:-1]])
        # Slice IDs and video codes
        sliced = self.data.loc[(indexer['IDs'], indexer['video_codes']), :]
        # Slice bouts
        bout_index_values = sliced.index.get_level_values('bout_index')
        for key in ('bout_idxs', 'idxs', 'bout_index', 'idx'):
            take_indices = kwargs.get(key)
            if take_indices is not None:
                if key[0] == 'i':
                    take_indices = bout_index_values.unique()[take_indices]
                sliced = sliced[np.isin(bout_index_values, take_indices)]
                break
        return sliced

    @staticmethod
    def _to_list(data, values=False, ndims=-1):
        if values:
            if ndims > 0:
                return [bout.reset_index(level=['ID', 'video_code', 'bout_index'], drop=True).values[:, :ndims] for
                        bout_index, bout in
                        data.groupby('bout_index')]
            else:
                return [bout.reset_index(level=['ID', 'video_code', 'bout_index'], drop=True).values for
                        bout_index, bout in
                        data.groupby('bout_index')]
        else:
            return [bout.reset_index(level=['ID', 'video_code', 'bout_index'], drop=True) for
                    bout_index, bout in
                    data.groupby('bout_index')]

    def get_bout(self, ID=None, video_code=None, bout_index=None, idx=0):
        bout = self._slice_data(ID=ID, video_code=video_code, bout_index=bout_index, idx=idx)
        bout.reset_index(level=('ID', 'video_code', 'bout_index'), drop=True, inplace=True)
        return bout

    def list_bouts(self, IDs=None, video_codes=None, bout_idxs=None, idxs=None, values=False, ndims=-1):
        if all([IDs is None, video_codes is None, bout_idxs is None, idxs is None]):
            return self._to_list(self.data, values=values, ndims=ndims)
        else:
            sliced = self._slice_data(IDs=IDs, video_codes=video_codes, bout_idxs=bout_idxs, idxs=idxs)
            return self._to_list(sliced, values=values, ndims=ndims)

    def whiten(self, mean=None, std=None):
        whitened = whiten_data(self.data, mean=mean, std=std)
        return BoutData(whitened, metadata=self.metadata)

    def transform(self, whiten=True, mean=None, std=None):
        if whiten:
            data_to_transform = whiten_data(self.data, mean=mean, std=std)
        else:
            data_to_transform = self.data
        transformed, pca = transform_data(data_to_transform)
        transformed = pd.DataFrame(transformed, index=self.data.index)
        return BoutData(transformed, metadata=self.metadata), pca

    def map(self, vectors, whiten=True, mean=None, std=None):
        if whiten:
            data_to_map = whiten_data(self.data, mean=mean, std=std)
        else:
            data_to_map = self.data
        mapped = map_data(data_to_map, vectors)
        mapped = pd.DataFrame(mapped, index=self.data.index, columns=self.data.columns)
        return BoutData(mapped, metadata=self.metadata)
