from plotting import *
from plotting.colors import virtual_prey_colors
from matplotlib import gridspec
from behaviour_analysis.manage_files import get_files
import pandas as pd
from behaviour_analysis.analysis.eye_convergence import EyeConvergenceAnalyser
from behaviour_analysis.miscellaneous import find_contiguous
import numpy as np


example_fish_directory = 'D:\\DATA\\virtual_prey_capture\\data\\2019_09_07\\190907_f7'
control_trial_index = 2
test_trial_index = 2
control_time = (440, 455)
test_time = (329.8, 344.8)

# 475 pixel = 1.8 cm
scale = 1.8 / 475

if __name__ == "__main__":

    # Import data
    fish_files, fish_paths = get_files(example_fish_directory, return_paths=True)
    tracking = pd.read_csv(fish_paths[0], delimiter=';', index_col=0)
    stimulus = pd.read_csv(fish_paths[4], delimiter=';', index_col=0)
    estimator = pd.read_csv(fish_paths[1], delimiter=';', index_col=0)
    threshold = estimator.loc[0, 'threshold']
    metadata = pd.read_json(fish_paths[3])

    # Calculate eye convergence
    tracking['convergence'] = (tracking['right_angle'] - tracking['left_angle']).rolling(window=30,
                                                                                         center=True).median()
    tracking = tracking[~np.isnan(tracking['convergence'])]
    # Calculate velocity
    tracking['v'] = np.linalg.norm(tracking[['f0_vx', 'f0_vy']], axis=1)
    tracking['v'] = tracking['v'].rolling(window=3, center=True).median()
    tracking['v'] = tracking['v'].rolling(window=15, center=True).mean()

    # Split tracking data into control and test trials
    stimuli = metadata.loc['log', 'stimulus']
    trials = stimuli[1::2]
    # Test trials
    test_trials = [trial for trial in trials if trial['disappear']]
    test_trial_times = [(trial['t_start'], trial['t_stop']) for trial in test_trials]
    test_tracking = [tracking[(tracking['t'] > start) & (tracking['t'] < stop)] for (start, stop) in
                     test_trial_times]
    # Control trials
    control_trials = [trial for trial in trials if not trial['disappear']]
    control_trial_times = [(trial['t_start'], trial['t_stop']) for trial in control_trials]
    control_tracking = [tracking[(tracking['t'] > start) & (tracking['t'] < stop)] for (start, stop) in
                        control_trial_times]

    # Calculate the eye convergence threshold
    ECA = EyeConvergenceAnalyser(tracking['convergence'].values, default_threshold=threshold)
    ECA.kernel_density_estimation()
    threshold = ECA.find_convergence_threshold()

    # Find hunting events
    eyes_converged = tracking[tracking['convergence'] > threshold].index
    hunting_events = find_contiguous(eyes_converged, minsize=30, stepsize=100)
    hunting_events = [(event[0], event[-1]) for event in hunting_events]

    # Find bouts
    mean = tracking['v'].mean()
    bouts = tracking[tracking['v'] > mean].index
    bouts = find_contiguous(bouts, minsize=5, stepsize=30)
    bouts = [(frames[0], frames[-1]) for frames in bouts]

    # Control trial
    control_trial = control_tracking[control_trial_index]
    control_trial = control_trial[(control_trial['t'] > control_time[0]) & (control_trial['t'] < control_time[1])]
    control_hunts = [event for event in hunting_events if (control_trial.index[0] < event[0] < control_trial.index[-1])]
    control_bouts = [bout for bout in bouts if (control_trial.index[0] < bout[0] < control_trial.index[-1])]
    control_stim = stimulus[(stimulus['t'] > control_time[0]) & (stimulus['t'] < control_time[1])]

    # Test trial
    test_trial = test_tracking[test_trial_index]
    test_trial = test_trial[(test_trial['t'] > test_time[0]) & (test_trial['t'] < test_time[1])]
    test_hunts = [event for event in hunting_events if (test_trial.index[0] < event[0] < test_trial.index[-1])]
    test_bouts = [bout for bout in bouts if (test_trial.index[0] < bout[0] < test_trial.index[-1])]
    test_stim = stimulus[(stimulus['t'] > test_time[0]) & (stimulus['t'] < test_time[1])]

    # Plot
    fig = plt.figure(figsize=(1.96, 3.54))
    gs = gridspec.GridSpec(2, 1, left=0, right=1, bottom=0, top=1)

    control_color = virtual_prey_colors['control']
    test_color = virtual_prey_colors['test']

    for i, (color, trial, hunts, bouts, stim) in enumerate([(control_color, control_trial, control_hunts, control_bouts, control_stim),
                                                             (test_color, test_trial, test_hunts, test_bouts, test_stim)]):
        subplt = gridspec.GridSpecFromSubplotSpec(2, 1, gs[i], wspace=0, hspace=0)
        ax1 = fig.add_subplot(subplt[0])
        ax2 = fig.add_subplot(subplt[1])

        t = trial.t.values
        dt = (t[-1] - t[0]) / len(t)

        # Plot eye convergence
        ax1.plot(t, trial.convergence, c='k', lw=1)
        # Plot hunting events
        for event in hunts:
            ax1.fill_between(trial['t'].loc[[event[0], event[-1]]], [0, 0], [85, 85], facecolor=color, alpha=0.5)
        # Plot stimulus
        stim_on = np.where(stim.show.values)[0]
        stim_on = find_contiguous(stim_on)
        for on in stim_on:
            ax1.plot(stim.iloc[on]['t'], stim.iloc[on]['show'] * 100, c=color, lw=2)

        # Plot velocity
        ax2.plot(t, trial.v * scale / dt, c='k', lw=1)
        ax2.plot([t[100], t[100]], [0, 1], c='k')
        # Plot bouts
        bout_onsets = [bout[0] for bout in bouts]
        bout_offsets = [bout[-1] for bout in bouts]
        is_hunting = []
        for offset in bout_offsets:
            if any([hunt[0] < offset < hunt[-1] for hunt in hunts]):
                is_hunting.append(1)
            else:
                is_hunting.append(0)
        fill_color = np.array(['w', color])
        ax2.scatter(trial.loc[bout_onsets].t, np.ones((len(bouts),)) * 3.5,
                    c=fill_color[is_hunting], edgecolor=color, lw=0.5, marker='d', s=10)

        # Axes
        ax1.set_xlim(t[0], t[-1])
        ax1.set_ylim(0, 100)
        ax1.axis('off')
        ax2.set_xlim(t[0], t[-1])
        ax2.set_ylim(0, 4 * scale / dt)
        ax2.axis('off')

    # plt.show()
    save_fig(fig, 'figure4', 'virtual_prey_example')
