from plotting import *
from behaviour_analysis.video import Video, video_code_to_timestamp
from behaviour_analysis.tracking import rotate_and_centre_image
from behaviour_analysis.manage_files import create_folder
from behaviour_analysis.miscellaneous import read_csv
from ast import literal_eval
import cv2
from skimage.exposure import rescale_intensity
import pandas as pd
import numpy as np
from datasets.main_dataset import experiment


exemplar_strikes = pd.read_csv(os.path.join(output_directory, 'figure5', 'exemplar_strikes.csv'),
                               index_col='bout_index', dtype={'ID': str, 'video_code': str})
attacks = exemplar_strikes.iloc[:6]
sstrikes = exemplar_strikes.iloc[10:16]
strike_video_directory = create_folder(output_directory, 'video4')

for strike_name, strike_cluster, pad in zip(('example_attacks', 'example_sstrikes'), (attacks, sstrikes), (50, 0)):
    save_images_to = create_folder(strike_video_directory, strike_name)
    panels = []
    for idx, bout_info in strike_cluster.iterrows():
        fish_info = experiment.data[experiment.data['ID'] == bout_info.ID].iloc[0]
        video_file = video_code_to_timestamp(bout_info.video_code)
        video_path = os.path.join(experiment.video_directory, fish_info.video_directory, video_file + '.avi')
        tracking_path = os.path.join(experiment.subdirs['tracking'], fish_info.ID, bout_info.video_code + '.csv')
        tracking = read_csv(tracking_path, centre=literal_eval)
        v = Video(video_path)
        frames = v.return_frames(bout_info.start, bout_info.end + pad)
        centered_frames = []
        for f, frame in zip(np.arange(bout_info.start, bout_info.end + pad), frames):
            tracking_info = tracking.loc[f, ['centre', 'heading']].to_dict()
            img = frame[..., 0]
            img = rescale_intensity(img, (50, img.max()))
            h, w = img.shape
            img = rotate_and_centre_image(img, np.array(tracking_info['centre']) / 2., tracking_info['heading'])
            cropped = img[h / 3: 2 * h / 3, w / 4: 2 * w / 3]
            scaled = cv2.resize(cropped, None, fx=2, fy=2).T[::-1]
            centered_frames.append(scaled)
        panels.append(centered_frames)
    shortest = min([len(panel) for panel in panels])
    panels = np.array([panel[:shortest] for panel in panels])
    concat = [np.concatenate(panels[i:i+3], axis=2) for i in range(0, 6, 3)]
    concat = np.concatenate(concat, axis=1)

    # for f in concat:
    #     cv2.imshow(strike_name, f)
    #     cv2.waitKey(200)
    # cv2.destroyAllWindows()

    for i, frame in enumerate(concat):
        output_path = os.path.join(save_images_to, str(i) + '.tiff')
        cv2.imwrite(output_path, frame)
    blank_frame = np.zeros(concat[0].shape, dtype='uint8')
    cv2.imwrite(os.path.join(save_images_to, 'blank.tiff'), blank_frame)
