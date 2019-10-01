from datasets.virtual_prey import fish_data, analysis_directory
from behaviour_analysis.analysis.eye_convergence import EyeConvergenceAnalyser
from behaviour_analysis.miscellaneous import find_contiguous, print_heading, print_subheading
import numpy as np
import os


if __name__ == "__main__":

    # =========
    # PREP DATA
    # =========

    print_heading('Prepping data')

    for key, data in fish_data.iteritems():

        print key

        # Calculate eye convergence as 100 ms rolling median of difference in eye angles
        tracking = data['tracking']
        tracking['convergence'] = (tracking['right_angle'] - tracking['left_angle']).rolling(window=30,
                                                                                             center=True).median()
        tracking = tracking[~np.isnan(tracking['convergence'])]
        tracking['dt'] = tracking['t'].diff()

        # Calculate velocity
        tracking['v'] = np.linalg.norm(tracking[['f0_vx', 'f0_vy']], axis=1)
        tracking['v'] = tracking['v'].rolling(window=3, center=True).median()
        tracking['v'] = tracking['v'].rolling(window=15, center=True).mean()

        # Update the tracking data frame
        data['tracking'] = tracking

        # Split tracking data into control, test and inter-stimulus trials
        stimuli = data['metadata'].loc['log', 'stimulus']
        pause = stimuli[0]
        trials = stimuli[1::2]
        inter_trials = stimuli[2::2]
        test_trials = [trial for trial in trials if trial['disappear']]
        control_trials = [trial for trial in trials if not trial['disappear']]

        inter_trial_times = [(trial['t_start'], trial['t_stop']) for trial in inter_trials]
        test_trial_times = [(trial['t_start'], trial['t_stop']) for trial in test_trials]
        control_trial_times = [(trial['t_start'], trial['t_stop']) for trial in control_trials]

        inter_tracking = [tracking[(tracking['t'] > start) & (tracking['t'] < stop)] for (start, stop) in
                          inter_trial_times]
        test_tracking = [tracking[(tracking['t'] > start) & (tracking['t'] < stop)] for (start, stop) in
                         test_trial_times]
        control_tracking = [tracking[(tracking['t'] > start) & (tracking['t'] < stop)] for (start, stop) in
                            control_trial_times]

        data['inter'] = inter_tracking
        data['test'] = test_tracking
        data['control'] = control_tracking

    print ''

    # ===========
    # FILTER DATA
    # ===========

    print_heading('Filtering data')

    keep_data = {}

    for key, data in fish_data.iteritems():

        print_subheading(key)

        # Calculate the eye convergence threshold
        ECA = EyeConvergenceAnalyser(data['tracking']['convergence'].values, default_threshold=data['threshold'])
        ECA.kernel_density_estimation()
        ECA.find_convergence_threshold()
        data['threshold'] = ECA.threshold
        print ''

        # Calculate prey capture score for each trial type
        for trial_type in ('control', 'test', 'inter'):
            times = []
            pc_times = []
            for trial in data[trial_type]:
                pc = trial[trial['convergence'] > data['threshold']]
                total_time = trial['dt'].sum()
                pc_time = pc['dt'].sum()
                times.append(total_time)
                pc_times.append(pc_time)
            total_time = sum(times)
            pc_time = sum(pc_times)
            data['{}_score'.format(trial_type)] = pc_time / total_time

        # Only keep animals whose prey capture score > 0.05 for control trials
        if data['control_score'] > 0.05:
            keep_data[key] = data

    n_analysed = len(keep_data)
    n_total = len(fish_data)
    percent = 100 * n_analysed / float(n_total)
    print 'Kept {} / {} animals ({}%) for further analysis\n'.format(n_analysed, n_total, percent)

    # ======================
    # ANALYSE HUNT DURATIONS
    # ======================

    print_heading('Analysing hunt durations')

    duration_control_averages = []
    duration_test_averages = []
    duration_inter_averages = []

    duration_control_all = []
    duration_test_all = []
    duration_inter_all = []

    for key, data in keep_data.iteritems():

        # Find hunting events
        eyes_converged = data['tracking'][data['tracking']['convergence'] > data['threshold']].index
        hunting_events = find_contiguous(eyes_converged, minsize=30, stepsize=100)
        data['hunting_events'] = [(event[0], event[-1]) for event in hunting_events]

        # Calculate hunt durations for each trial type
        hunt_durations = {}
        for trial_type in ('control', 'test', 'inter'):
            trial_type_durations = []
            for trial in data[trial_type]:
                trial_hunts = [event for event in data['hunting_events'] if
                               (trial.index[0] <= event[0] < trial.index[-1])]
                for hunt in trial_hunts:
                    duration = data['tracking'].loc[hunt[-1], 't'] - trial.loc[hunt[0], 't']
                    trial_type_durations.append(duration)
            hunt_durations[trial_type] = trial_type_durations
        data['hunt_durations'] = hunt_durations

        # Calculate average per fish
        duration_control_averages.append(np.nanmean(hunt_durations['control']))
        duration_test_averages.append(np.nanmean(hunt_durations['test']))
        duration_inter_averages.append(np.nanmean(hunt_durations['inter']))

        # Add fish data to total
        duration_control_all.extend(hunt_durations['control'])
        duration_test_all.extend(hunt_durations['test'])
        duration_inter_all.extend(hunt_durations['inter'])

    # Save average fish hunt durations as a (n_fish x 3) array
    fish_hunt_durations = np.array([duration_control_averages, duration_test_averages, duration_inter_averages]).T
    np.save(os.path.join(analysis_directory, 'fish_hunt_durations.npy'), fish_hunt_durations)

    # Save hunt duration distributions
    np.save(os.path.join(analysis_directory, 'control_durations.npy'), duration_control_all)
    np.save(os.path.join(analysis_directory, 'test_durations.npy'), duration_test_all)
    np.save(os.path.join(analysis_directory, 'inter_durations.npy'), duration_inter_all)

    print ''

    # =============
    # ANALYSE BOUTS
    # =============

    print_heading('Analysing bouts')

    sequence_control_averages = []
    sequence_test_averages = []
    sequence_inter_averages = []

    sequence_control_all = []
    sequence_test_all = []
    sequence_inter_all = []

    for key, data in keep_data.iteritems():

        # Find bouts
        mean = data['tracking']['v'].mean()
        std = data['tracking']['v'].std()
        bout_frames = data['tracking'][data['tracking']['v'] > mean].index
        bout_frames = find_contiguous(bout_frames, minsize=5, stepsize=30)
        bout_frames = [(frames[0], frames[-1]) for frames in bout_frames]
        data['bouts'] = bout_frames
        print 'Fish {} bout rate: {}'.format(key, len(bout_frames) / 360.)

        # Identify hunting bout sequences in each trial type
        bout_sequences = {}
        for trial_type in ('control', 'test', 'inter'):
            trial_type_hunts = []
            for trial in data[trial_type]:
                trial_hunts = [event for event in data['hunting_events'] if
                               (trial.index[0] <= event[0] < trial.index[-1])]
                trial_bouts = [bout for bout in data['bouts'] if (trial.index[0] <= bout[0] < trial.index[-1])]
                for hunt in trial_hunts:
                    hunt_bouts = [bout for bout in trial_bouts if (hunt[0] < bout[-1] < hunt[-1])]
                    trial_type_hunts.append(len(hunt_bouts))
            bout_sequences[trial_type] = trial_type_hunts
        data['bout_sequences'] = bout_sequences

        # Find median sequence length per fish
        sequence_control_averages.append(np.mean(bout_sequences['control']))
        sequence_test_averages.append(np.mean(bout_sequences['test']))
        sequence_inter_averages.append(np.mean(bout_sequences['inter']))

        # Add sequence lengths to distributions
        sequence_control_all.extend(bout_sequences['control'])
        sequence_test_all.extend(bout_sequences['test'])
        sequence_inter_all.extend(bout_sequences['inter'])

    # Save median fish sequence lengths as a (n_fish x 3) array, replacing nans with 0
    fish_sequence_lengths = np.array([sequence_control_averages, sequence_test_averages, sequence_inter_averages]).T
    fish_sequence_lengths[np.isnan(fish_sequence_lengths)] = 0
    np.save(os.path.join(analysis_directory, 'fish_sequence_lengths.npy'), fish_sequence_lengths)

    # Save sequence length distributions
    np.save(os.path.join(analysis_directory, 'control_sequences.npy'), sequence_control_all)
    np.save(os.path.join(analysis_directory, 'test_sequences.npy'), sequence_test_all)
    np.save(os.path.join(analysis_directory, 'inter_sequences.npy'), sequence_inter_all)
