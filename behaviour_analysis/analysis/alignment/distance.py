import numpy as np
from joblib import Parallel, delayed

from .dynamic_time_warping import fill_row, fill_row_min, fill_row_1d


def calculate_distance_matrix_(bouts, templates=None, fs=500., bw=0.01, flip=False, fill_function='auto', parallel_processing=True, n_processors=-1):

    if fill_function == 'auto':
        fill_func = fill_row
    elif fill_function == 'min':
        fill_func = fill_row_min
    else:
        raise ValueError('fill_function must be: {auto, min}')

    bouts = list(bouts)
    n_rows = len(bouts)

    if templates is not None:
        use_templates = True
        templates = list(templates)
        bouts_by_row = [[bout] + templates for bout in bouts]
    else:
        use_templates = False
        n_columns = n_rows
        bouts_by_row = [bouts[i:] for i in range(n_columns - 1)]

    if parallel_processing:
        distances = Parallel(n_processors)(delayed(fill_func)(*row, fs=fs, bw=bw, flip=flip) for row in bouts_by_row)
    else:
        distances = [fill_func(*row, fs=fs, bw=bw, flip=flip) for row in bouts_by_row]

    if use_templates:
        D = np.array(distances)
    else:
        D = np.array([d for row in distances for d in row])

    return D


def calculate_distance_matrix_templates(bouts, templates, fs=500., bw=0.01, parallel_processing=True, n_processors=-1):
    bouts = list(bouts)
    templates = list(templates)
    bouts_by_row = [[bout] + templates for bout in bouts]
    if parallel_processing:
        distances = Parallel(n_processors)(delayed(fill_row_min)(*row, fs=fs, bw=bw) for row in bouts_by_row)
    else:
        distances = [fill_row_min(*row, fs=fs, bw=bw) for row in bouts_by_row]
    D = np.array(distances)
    return D


def calculate_distance_matrix(bouts, fs=500., bw=0.01, flip=False, parallel_processing=True, n_processors=-1):

    is_1d = (bouts[0].ndim == 1)

    bouts = list(bouts)
    bouts_by_row = [bouts[i:] for i in range(len(bouts) - 1)]

    if parallel_processing:
        if is_1d:
            distances = Parallel(n_processors)(delayed(fill_row_1d)(*row, fs=fs, bw=bw)
                                               for row in bouts_by_row)
        else:
            distances = Parallel(n_processors)(delayed(fill_row)(*row, fs=fs, bw=bw, flip=flip)
                                               for row in bouts_by_row)
    else:
        if is_1d:
            distances = [fill_row_1d(*row, fs=fs, bw=bw) for row in bouts_by_row]
        else:
            distances = [fill_row(*row, fs=fs, bw=bw, flip=flip) for row in bouts_by_row]

    D = np.array([d for row in distances for d in row])
    return D
