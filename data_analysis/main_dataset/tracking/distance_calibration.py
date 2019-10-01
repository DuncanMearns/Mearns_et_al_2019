from datasets.main_dataset import experiment
import numpy as np

ROI_lengths = []
for idx, fish_info in experiment.data.iterrows():
    (x0, y0), (x1, y1) = fish_info.ROI
    ROI_lengths.append((y1 - y0))
    ROI_lengths.append((x1 - x0))

average_pixel_number = np.mean(ROI_lengths)
chamber_size = 15  # mm
mm_per_pixel = chamber_size / average_pixel_number
print mm_per_pixel
