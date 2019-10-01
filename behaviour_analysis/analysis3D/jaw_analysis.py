from ..analysis import transform_data, whiten_data, map_data
from ..miscellaneous import print_heading
import os
import numpy as np
import pandas as pd


def open_kinematics(events, kinematics_directory):
    # =========================
    # IMPORT ALL KINEMATIC DATA
    # =========================
    print_heading('IMPORTING DATA')
    # Initialise lists for storing DataFrames and indices
    kinematics_dfs = []
    multi_index_keys = []
    # Iterate through fish IDs
    for ID, fish_events in events.groupby('ID'):
        print ID
        print '-' * len(ID)
        # Iterate through video codes
        for video_code, video_events in fish_events.groupby('video_code'):
            print video_code
            # Open kinematics DataFrame
            kinematics_path = os.path.join(kinematics_directory, ID, video_code + '.csv')
            df = pd.read_csv(kinematics_path)
            # Assign frames to bouts
            event_indexer = np.zeros((len(df),), dtype='i4') - 1
            for idx, event_info in video_events.iterrows():
                event_indexer[event_info.start:event_info.end] = idx
            event_indexer = zip(event_indexer, df.index)
            df.index = pd.MultiIndex.from_tuples(event_indexer, names=['event_index', 'frame'])
            # Append DataFrame and indices to lists
            kinematics_dfs.append(df)
            multi_index_keys.append((ID, video_code))
        print ''
    # Create the multi-indexed data attribute
    data = pd.concat(kinematics_dfs, keys=multi_index_keys, names=['ID', 'video_code'])
    return data


class KinematicData(object):

    def __init__(self, data, metadata):
        self.data = data
        assert isinstance(self.data.index, pd.core.index.MultiIndex), 'Data must be MultiIndexed!'
        self.metadata = metadata

    @property
    def mean(self):
        return self.data.mean(axis=0)

    @property
    def std(self):
        return self.data.std(axis=0)

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
        bout_index_values = sliced.index.get_level_values('event_index')
        for key in ('event_idxs', 'idxs', 'event_index', 'idx'):
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
                return [bout.reset_index(level=['ID', 'video_code', 'event_index'], drop=True).values[:, :ndims] for
                        bout_index, bout in
                        data.groupby('event_index')]
            else:
                return [bout.reset_index(level=['ID', 'video_code', 'event_index'], drop=True).values for
                        bout_index, bout in
                        data.groupby('event_index')]
        else:
            return [bout.reset_index(level=['ID', 'video_code', 'event_index'], drop=True) for
                    bout_index, bout in
                    data.groupby('event_index')]

    def get_event(self, ID=None, video_code=None, event_index=None, idx=0):
        bout = self._slice_data(ID=ID, video_code=video_code, bout_index=event_index, idx=idx)
        bout.reset_index(level=('ID', 'video_code', 'event_index'), drop=True, inplace=True)
        return bout

    def list_events(self, IDs=None, video_codes=None, event_idxs=None, idxs=None, values=False, ndims=-1):
        if all([IDs is None, video_codes is None, event_idxs is None, idxs is None]):
            return self._to_list(self.data, values=values, ndims=ndims)
        else:
            sliced = self._slice_data(IDs=IDs, video_codes=video_codes, event_idxs=event_idxs, idxs=idxs)
            return self._to_list(sliced, values=values, ndims=ndims)

    def whiten(self, mean=None, std=None):
        whitened = whiten_data(self.data, mean=mean, std=std)
        return KinematicData(whitened, metadata=self.metadata)

    def transform(self, whiten=True, mean=None, std=None):
        if whiten:
            data_to_transform = whiten_data(self.data, mean=mean, std=std)
        else:
            data_to_transform = self.data
        transformed, pca = transform_data(data_to_transform)
        transformed = pd.DataFrame(transformed, index=self.data.index)
        return KinematicData(transformed, metadata=self.metadata), pca

    def map(self, vectors, whiten=True, mean=None, std=None):
        if whiten:
            data_to_map = whiten_data(self.data, mean=mean, std=std)
        else:
            data_to_map = self.data
        mapped = map_data(data_to_map, vectors)
        mapped = pd.DataFrame(mapped, index=self.data.index, columns=self.data.columns)
        return KinematicData(mapped, metadata=self.metadata)


class JawData(KinematicData):

    def __init__(self, data, metadata=None):
        KinematicData.__init__(self, data, metadata)

    @classmethod
    def from_directory(cls, events, kinematics_directory, check_magnitude=True, elevation=False):
        # Open the bouts DataFrame
        if type(events) == pd.DataFrame:
            events_df = events
        elif type(events) == str:
            events_df = pd.read_csv(events, dtype={'ID': str, 'video_code': str})
        else:
            raise TypeError('bouts must be string or DataFrame')
        # Import kinematics data
        data = open_kinematics(events_df, kinematics_directory)
        data = data[data.index.get_level_values('event_index').isin(events_df.index)]
        if elevation:
            data = data[['depression', 'elevation', 'fish_elevation']]
        else:
            data = data[['depression', 'elevation']]
        jaw_data = cls(data, events_df)
        if check_magnitude:
            print 'Checking jaw movement magnitudes...',
            whitened = jaw_data.whiten()
            keep_events = []
            for event_index, event in whitened.data.groupby('event_index'):
                if np.all(event.max() < 10):
                    keep_events.append(event_index)
            keep_metadata = jaw_data.metadata.loc[keep_events]
            keep_data = jaw_data.data[jaw_data.data.index.get_level_values('event_index').isin(keep_events)]
            jaw_data = cls(keep_data, keep_metadata)
            print 'done!'
        return jaw_data




# OLD CODE HERE

class BehaviourAnalysis3D(object):

    def __init__(self, experiment, bouts_file='bouts.csv'):

        self.experiment = experiment
        self.bouts_df = read_csv(os.path.join(self.experiment.directory, bouts_file), ID=str, video_code=str)

        print 'importing bouts...'
        self.bouts = []
        for ID, fish_bouts in self.bouts_df.groupby('ID'):
            print ID
            for video_code, video_bouts in fish_bouts.groupby('video_code'):
                print '\t' + video_code
                kinematics_path = os.path.join(self.experiment.directory, 'kinematics', ID, video_code + '.csv')
                kinematics = pd.read_csv(kinematics_path)
                for idx, bout_info in video_bouts.iterrows():
                    bout = kinematics.loc[bout_info.start:bout_info.end, :]
                    self.bouts.append(bout)
        assert len(self.bouts_df) == len(self.bouts)
        print 'done!'

        print 'finding jaw movements...',
        self._find_jaw_movements()
        print 'done!'

    def _transform_bouts(self):
        self.tail_columns = [col for col in self.bouts[0].columns if col[0] == 'k']
        self.frames = pd.concat(self.bouts, ignore_index=True).loc[:, self.tail_columns]
        self.mean, self.std = self.frames.mean(axis=0), self.frames.std(axis=0)
        self.whitened_frames = (self.frames - self.mean) / self.std
        self.whitened_bouts = [(bout.loc[:, self.tail_columns] - self.mean) / self.std for bout in self.bouts]
        self.pca = PCA()
        self.pca.fit(self.whitened_frames)
        self.transformed_bouts = [self.pca.transform(bout) for bout in self.whitened_bouts]

    def transform_bouts(self):
        self._transform_bouts()
        return self.transformed_bouts

    def _calculate_distance_matrix(self, ndims, fs, kinematics='tail'):
        if kinematics == 'tail':
            samples = [bout[:, :ndims] for bout in self.transformed_bouts]
        elif kinematics == 'jaw':
            samples = [bout[:, :ndims] for bout in self.transformed_jaw_movements]
        else:
            raise ValueError('key word kinematics must be: {tail, jaw}')
        n_samples = len(samples)
        print '===========================\n' \
              'CALCULATING DISTANCE MATRIX\n' \
              '===========================\n'
        D = np.zeros((n_samples, n_samples))
        bouts_by_row = [samples[i:] for i in range(n_samples - 1)]
        if kinematics == 'tail':
            rows = Parallel(-1)(delayed(fill_row_min)(*bouts, fs=fs) for bouts in bouts_by_row)
        elif kinematics == 'jaw':
            rows = Parallel(-1)(delayed(fill_row_1d)(*bouts, fs=fs) for bouts in bouts_by_row)
        for i, row in enumerate(rows):
            D[i, i + 1:] = row
            D[i + 1:, i] = row
        self.D = D

    def calculate_distance_matrix(self, kinematics='tail', ndims=3, fs=400., save=False, output_path=None):
        start_time = time.time()
        self._calculate_distance_matrix(ndims=ndims, fs=fs, kinematics=kinematics)
        end_time = time.time()
        if (end_time - start_time) / 60 > 60:
            print 'Time taken:', (end_time - start_time) / 3600, 'hours'
        else:
            print 'Time taken:', (end_time - start_time) / 60, 'minutes'
        if save:
            if output_path is None:
                output_path = 'distance_matrix.npy'
            np.save(output_path, self.D)
        return self.D

    def _open_distance_matrix(self, path):
        self.D = np.load(path)

    def open_distance_matrix(self, path):
        if os.path.exists(path):
            self._open_distance_matrix(path)
            return self.D
        else:
            raise ValueError('Path does not exist!')

    def _cluster_bouts(self, preference):
        if preference == 'minimum':
            diagonal = np.min(-self.D)
        elif preference == 'maximum':
            diagonal = np.max(-self.D)
        elif preference == 'median':
            diagonal = np.median(-self.D)
        else:
            raise ValueError('preference must be: {minimum, maximum, median}')
        preference = np.array([diagonal] * len(self.D))
        print '================\n' \
              'CLUSTERING BOUTS\n' \
              '================\n'
        self.cluster = AffinityPropagation(affinity='precomputed', preference=np.min(-self.D))
        self.cluster.fit_predict(-self.D)
        self.n_clusters = len(np.unique(self.cluster.labels_))

    def cluster_bouts(self, preference='median'):
        self._cluster_bouts(preference)
        return self.cluster

    def _find_jaw_movements(self):

        tracked_segments = []
        for idx, fish_info in self.experiment.data.iterrows():
            print fish_info.ID
            kinematic_files, kinematic_paths = get_files(fish_info.kinematics_directory_)
            for p in kinematic_paths:
                kinematics_df = pd.read_csv(p)
                tracked_side = kinematics_df[kinematics_df['side_tracked']]
                tracked_segments.append(tracked_side.loc[:, ['depression', 'elevation', 'fish_elevation']])

        all_jaw_segments = []
        for segment in tracked_segments:
            jaw_segments = find_contiguous(segment.index)
            for frames in jaw_segments:
                all_jaw_segments.append(segment.loc[frames, ['depression', 'elevation']])

        all_jaw_frames = pd.concat(all_jaw_segments, ignore_index=True)
        threshold = all_jaw_frames['depression'].quantile()

        jaw_movements = []
        for segment in all_jaw_segments:
            noms = find_noms(segment, threshold)
            for frames in noms:
                jaw_movements.append(segment.loc[frames[0]:frames[1]])

        all_jaw_movement_frames = pd.concat(jaw_movements, ignore_index=True)
        mean_jaw, std_jaw = all_jaw_movement_frames.mean(axis=0), all_jaw_movement_frames.std(axis=0)
        whitened_jaw_movements = [(jaw_movement - mean_jaw) / std_jaw for jaw_movement in jaw_movements]

        keep_jaw_indices = [i for i, nom in enumerate(whitened_jaw_movements) if (nom.max(axis=0) < 10).all()]
        self.jaw_movements = [nom for i, nom in enumerate(jaw_movements) if i in keep_jaw_indices]

    def _transform_jaw_movements(self):
        all_jaw_movement_frames = pd.concat(self.jaw_movements, ignore_index=True)
        mean_jaw, std_jaw = all_jaw_movement_frames.mean(axis=0), all_jaw_movement_frames.std(axis=0)
        whitened_jaw_movement_frames = (all_jaw_movement_frames - mean_jaw) / std_jaw
        pca = PCA().fit(whitened_jaw_movement_frames)
        min_length = min([len(jaw_movement) for jaw_movement in self.jaw_movements])
        transformed_jaw_movements = [pca.transform((jaw_movement - mean_jaw) / std_jaw) for jaw_movement in self.jaw_movements]
        self.transformed_jaw_movements = [t[:min_length] for t in transformed_jaw_movements]

    def transform_jaw_movements(self):
        self._transform_jaw_movements()
        return self.transformed_jaw_movements


if __name__ == "__main__":

    experiment = TrackingExperiment3D('F:\\DATA\\3D_prey_capture')
    behaviour_analyser = BehaviourAnalysis3D(experiment)
    output_directory = create_folder(experiment.directory, 'bout_analysis')

    # behaviour_analyser.transform_bouts()
    # behaviour_analyser.calculate_distance_matrix(ndims=4, save=True, output_path=os.path.join(output_directory, 'distance_matrix.npy'))

    behaviour_analyser.transform_jaw_movements()
    behaviour_analyser.calculate_distance_matrix(kinematics='jaw', ndims=1, save=True, output_path=os.path.join(output_directory, 'jaw_distance_matrix.npy') )

    # behaviour_analyser._open_distance_matrix(os.path.join(output_directory, 'distance_matrix.npy'))
    # behaviour_analyser._cluster_bouts(preference='minimum')
    # print behaviour_analyser.n_clusters
    # np.save(os.path.join(output_directory, 'cluster_labels.npy'), behaviour_analyser.cluster.labels_)
    # np.save(os.path.join(output_directory, 'cluster_centres.npy'), behaviour_analyser.cluster.cluster_centers_indices_)