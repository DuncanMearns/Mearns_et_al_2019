from ..miscellaneous import find_contiguous
import numpy as np
import pandas as pd


def find_jaw_movements(jaw_df, threshold):
    smoothed_jaw = jaw_df['depression'].rolling(window=20, center=True, min_periods=0).mean()
    threshed = smoothed_jaw > threshold
    above_thresh = threshed[threshed]
    nom_frames = find_contiguous(above_thresh.index, minsize=40)
    if len(nom_frames) > 0:
        if nom_frames[0][0] == jaw_df.index[0]:
            nom_frames.pop(0)
        if len(nom_frames) > 0:
            if nom_frames[-1][-1] == jaw_df.index[-1]:
                nom_frames.pop(-1)

    all_noms = []
    for frames in nom_frames:
        post_nom = smoothed_jaw.loc[frames[-1]:]
        post_nom_diffed = post_nom.diff().apply(lambda x: np.sign(x))
        try:
            last_frame = post_nom_diffed[post_nom_diffed > 0].index[0]
        except IndexError:
            last_frame = post_nom_diffed.index[-1]
        pre_nom = smoothed_jaw.loc[:frames[0]]
        pre_nom_diffed = pre_nom.diff().apply(lambda x: np.sign(x))
        try:
            first_frame = pre_nom_diffed[pre_nom_diffed < 0].index[-1]
        except IndexError:
            first_frame = pre_nom_diffed.index[0]
        all_noms.append((first_frame, last_frame))

    return [(frames[0], frames[-1]) for frames in nom_frames]


def find_video_jaw_movements(kinematics):
    """Find bouts within a video using a given threshold and minimum bout length

    This function first finds continuously tracked sections of the video that do not contain any missing data. Next, it
    finds bouts within each continuously tracked segment using the find_bouts function with the given parameters. Then,
    it splits bouts longer than 400 ms into shorter bouts using the split_long_bout function. Finally, it returns the
    frame numbers of each bout that was detected in chronological order.

    Parameters
    ----------
    kinematics : pd.DataFrame or str
        Path to a .csv file containing kinematic data or pre-loaded kinematic data as a DataFrame
    fs : float
        The sampling frequency (frames per second)
    threshold : float, optional (default = 0.02)
        Threshold used for finding bouts
    min_length : float, optional (default = 0.1)
        Minimum bout length (seconds)

    Returns
    -------
    video_bout_frames : list
        A list of tuples containing the first and last frame of each bout

    See Also
    --------
    find_contiguous
    find_bouts
    split_long_bout
    SetBoutDetectionThreshold
    """

    if type(kinematics) == str:
        kinematics_df = pd.read_csv(kinematics)
    else:
        kinematics_df = kinematics

    tracked_side = kinematics_df[kinematics_df['side_tracked']]
    tracked_segments = find_contiguous(tracked_side.index)
    jaw_segments = []
    for frames in tracked_segments:
        jaw_segments.append(tracked_side.loc[frames, ['depression', 'elevation']])

    jaw_frames = pd.concat(jaw_segments, ignore_index=True)
    threshold = jaw_frames['depression'].quantile()

    jaw_movement_frames = []
    for segment in jaw_segments:
        noms = find_jaw_movements(segment, threshold)
        jaw_movement_frames += noms

    return jaw_movement_frames
