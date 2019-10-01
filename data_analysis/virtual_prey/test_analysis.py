from datasets.virtual_prey import fish_data
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

for key, data in fish_data.iteritems():
    print key

    tracking = data['tracking']
    tracking['convergence'] = tracking['right_angle'] - tracking['left_angle']
    tracking = tracking[~np.isnan(tracking['convergence'])]

    stimuli = data['metadata'].loc['log', 'stimulus']
    pause = stimuli[0]
    trials = stimuli[1::2]
    inter_trials = stimuli[2::2]
    test_trials = [trial for trial in trials if trial['disappear']]
    control_trials = [trial for trial in trials if not trial['disappear']]

    inter_trial_times = [(trial['t_start'], trial['t_stop']) for trial in inter_trials]
    test_trial_times = [(trial['t_start'], trial['t_stop']) for trial in test_trials]
    control_trial_times = [(trial['t_start'], trial['t_stop']) for trial in control_trials]

    inter_tracking = pd.concat([tracking[(tracking['t'] > start) & (tracking['t'] < stop)]
                                     for (start, stop) in inter_trial_times], axis=0)
    test_tracking = pd.concat([tracking[(tracking['t'] > start) & (tracking['t'] < stop)]
                                    for (start, stop) in test_trial_times], axis=0)
    control_tracking = pd.concat([tracking[(tracking['t'] > start) & (tracking['t'] < stop)]
                                       for (start, stop) in control_trial_times], axis=0)

    print data['threshold']

    plt.figure()
    plt.hist(inter_tracking['convergence'], bins=np.arange(-20, 80), normed=True, alpha=0.5)
    plt.hist(test_tracking['convergence'], bins=np.arange(-20, 80), normed=True, alpha=0.5)
    plt.hist(control_tracking['convergence'], bins=np.arange(-20, 80), normed=True, alpha=0.5)
    plt.show()

    # for t1, t2 in zip(times, times[1:]):
    #     seg = tracking[(tracking['t'] > t1) & (tracking['t'] < t2)]
    #     plt.figure()
    #     plt.hist(seg['convergence'], bins=np.arange(-20, 80), normed=True)
    # plt.show()
