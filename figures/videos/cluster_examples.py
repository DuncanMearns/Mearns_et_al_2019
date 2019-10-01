from plotting import *
from datasets.main_dataset import experiment
from behaviour_analysis.video import Video, video_code_to_timestamp
from behaviour_analysis.miscellaneous import read_csv
from behaviour_analysis.tracking import rotate_and_centre_image
from behaviour_analysis.manage_files import create_folder
import numpy as np
import pandas as pd
import os
from ast import literal_eval
import cv2
from skimage.exposure import rescale_intensity


if __name__ == "__main__":

    video_output = create_folder(output_directory, 'video3')

    representative_bouts = pd.read_csv(os.path.join(experiment.subdirs['analysis'], 'clustering',
                                                    'representative_bouts.csv'),
                                      index_col='bout_index', dtype={'ID': str, 'video_code': str})

    for idx, bout_info in representative_bouts.iterrows():

        fish_info = experiment.data[experiment.data['ID'] == bout_info.ID].iloc[0]
        video_file = video_code_to_timestamp(bout_info.video_code)
        video_path = os.path.join(experiment.video_directory, fish_info.video_directory, video_file + '.avi')
        tracking_path = os.path.join(experiment.subdirs['tracking'], fish_info.ID, bout_info.video_code + '.csv')
        tracking = read_csv(tracking_path, centre=literal_eval)

        v = Video(video_path)
        frames = v.return_frames(bout_info.start, bout_info.end)

        bout_output = create_folder(video_output, 'cluster_{}'.format(bout_info.module))

        for f, frame in zip(np.arange(bout_info.start, bout_info.end), frames):
            tracking_info = tracking.loc[f, ['centre', 'heading']].to_dict()
            img = frame[..., 0]
            img = rescale_intensity(img, (50, img.max()))
            h, w = img.shape
            img = rotate_and_centre_image(img, np.array(tracking_info['centre']) / 2., tracking_info['heading'])
            cropped = img[h / 3: 2 * h / 3, w / 4: 2 * w / 3]
            scaled = cv2.resize(cropped, None, fx=2, fy=2).T[::-1]
            cv2.imwrite(os.path.join(bout_output, str(f)+'.tiff'), scaled)

    blank = np.zeros(scaled.shape, dtype='uint8')
    cv2.imwrite(os.path.join(video_output, 'blank.tiff'), blank)
