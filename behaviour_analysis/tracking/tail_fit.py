import numpy as np
from skimage.morphology import skeletonize


def fit_tail(mask, centre, opposite_vector, n_points):
    """Fits ordered points to a skeletonised image

    Parameters
    ----------
    mask : array
        Array representing the mask of the fish in an image
    centre : tuple or list or array-like (len 2)
        The (x, y) coordinate of the starting point. The nearest point in the skeleton will be the first tail point
    opposite_vector : tuple or list or array-like (len 2)
        The (dx, dy) vector representing the heading of the fish
    n_points : int
        The number of points to fit to the tail

    Returns
    -------
    tail_points : array like, shape (n_points, 2)
        Ordered points along the head-tail axis of the fish
    """
    skeleton = skeletonize(mask.astype('uint8'))
    try:
        skeleton_points = fast_longestpath(skeleton, np.array(centre))
        skeleton_points = skeleton_points[:, ::-1]
    except Exception:
        skeleton_paths = walk_skeleton(skeleton, np.array(centre), first_walk=True)
        path_directions = []
        for skeleton_path in skeleton_paths:
            path_vectors = np.diff(skeleton_path[:20, ::-1], axis=0)
            dots = [np.dot(v / np.linalg.norm(v), opposite_vector) for v in path_vectors]
            mean_dot = np.mean(np.array(dots))
            path_directions.append(mean_dot)
        skeleton_points = skeleton_paths[path_directions.index(min(path_directions))][:, ::-1]
    if n_points == -1:
        return skeleton_points
    else:
        current_indices = np.arange(len(skeleton_points))
        interpolate_indices = np.linspace(0, len(skeleton_points) - 1, n_points)
        new_x = np.interp(interpolate_indices, current_indices, skeleton_points[:, 0])
        new_y = np.interp(interpolate_indices, current_indices, skeleton_points[:, 1])
        tail_points = np.array([new_x, new_y]).T
        return tail_points


def walk_skeleton(skeleton_image, seed_point, first_walk=False):
    """Finds the longest path or paths from a seed point to the end of a skeleton

    This is a recursive function. When called for the first time, first_walk should be set to True.

    Parameters
    ----------
    skeleton_image : array
        Boolean array of a skeletonised image
    seed_point : array (shape: (2,))
        Starting point for finding paths
    first_walk : bool, default False
        If the function is called outside of itself, first_walk should be True

    Returns
    -------
    paths : list
        If first_walk == True, returns the longest path to an endpoint of the skeleton starting from the seed point
        for each starting direction (list of arrays).
        If the function is called recursively, returns the longest path to an end point (array).
    """
    # create masks for selecting neighbouring points
    row_mask = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    col_mask = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    if first_walk:
        # pad array with zeros to handle edges and corners
        padded_skeleton = np.pad(skeleton_image, ((1, 1), (1, 1)), 'constant', constant_values=0)
    else:
        padded_skeleton = skeleton_image
    # find all points in the image
    skeleton_points = np.argwhere(padded_skeleton)
    # find the distance of each point from the seed and find the index of the nearest point to the seed
    if first_walk:
        padded_seed = seed_point + np.array([1, 1])
        delta_vector = skeleton_points - padded_seed[::-1]
        distance_squared = np.sum(delta_vector ** 2, axis=1)
        centre_index = np.argmin(distance_squared)
        # find the first point in the path, set it as the current point and create the path
        start_walk = skeleton_points[centre_index]
    else:
        start_walk = seed_point[::-1]
    current_point = start_walk.copy()
    path = np.expand_dims(current_point, axis=0)
    # do not terminate the walk until an end point has been reached or all branches have been explored
    if first_walk:
        all_possible_paths = []
    terminate = False
    while not terminate:
        # remove the current point from the array
        padded_skeleton[current_point[0], current_point[1]] = 0
        # find all neighbouring points to the current point in the array using row and columns masks
        window = padded_skeleton[row_mask + current_point[0], col_mask + current_point[1]]
        # find all possible vectors that move to a neighbouring point
        steps = np.argwhere(window) - np.array([1, 1])
        n_steps = steps.shape[0]
        # if there are no vectors that move to a point in the array, an end point has been reached
        if n_steps == 0:
            if first_walk:
                all_possible_paths.append(path)
            terminate = True
        # if there is only one neighbouring point, it is added to the path and becomes the current point
        elif n_steps == 1:
            current_point = current_point.copy() + steps.squeeze()
            path = np.concatenate((path, np.expand_dims(current_point, axis=0)), axis=0)
        # if there are more than one neighbouring points, the path could continue in multiple directions
        else:
            # branch points are possible paths to continue the walk
            branch_points = steps + current_point.copy()
            # remove branch points from the array
            padded_skeleton[branch_points[:, 0], branch_points[:, 1]] = 0
            # each branch point becomes a seed for a new path (branch of the main path)
            branch_seeds = branch_points[:, ::-1]
            branches = [walk_skeleton(padded_skeleton, seed) for seed in branch_seeds]
            branch_lengths = [branch.shape[0] for branch in branches]
            if first_walk:
                for i, branch in enumerate(branches):
                    possible_path = np.concatenate((path, branch), axis=0)
                    all_possible_paths.append(possible_path)
            else:
                if not np.all(np.array(branch_lengths) == 0):
                    longest_i = branch_lengths.index(max(branch_lengths))
                    longest_branch = branches[longest_i]
                    path = np.concatenate((path, longest_branch), axis=0)
            # all paths have been explored so the walk can be terminated
            terminate = True
    # return the path
    if first_walk:
        all_possible_paths = [path - np.array([1, 1]) for path in all_possible_paths]
        return all_possible_paths
    else:
        return path


shifts = np.asarray(((-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1),))

def fast_longestpath(skeleton_image, seed_point):
    mod_skel_image = skeleton_image.copy()  # Copy of image to 'eat' as pathing progresses
    direction_map = np.zeros(mod_skel_image.shape + (shifts.shape[0],), dtype=np.bool)  # for tracing back at the end

    skeleton_points = np.argwhere(mod_skel_image)
    delta_vector = skeleton_points - seed_point[::-1]
    distance_squared = np.sum(delta_vector ** 2, axis=1)
    centre_index = np.argmin(distance_squared)
    # find the first point in the path, set it as the current point and create the path
    start_walk = skeleton_points[centre_index]

    current_pointsx = np.asarray(start_walk[0], dtype=np.int)
    current_pointsy = np.asarray(start_walk[1], dtype=np.int)

    for iteration in range(2000):
        newpointsx = []
        newpointsy = []

        # shift each direction
        for i in range(len(shifts)):
            indx, indy = current_pointsx + shifts[i, 0], current_pointsy + shifts[i, 1]
            # do we find a 'hit' in this new shift direction for any of the current points?
            hits = mod_skel_image[indx, indy]

            if hits.any():
                # Remove hits from image
                mod_skel_image[indx[hits], indy[hits]] = False
                # Add our current shift to the direction map
                direction_map[indx[hits], indy[hits], i] = True
                newpointsx.append(indx[hits])
                newpointsy.append(indy[hits])

        if len(newpointsx) == 0:
            break
        current_pointsx = np.hstack(newpointsx)
        current_pointsy = np.hstack(newpointsy)

    # Done, have to trace back the path
    path = np.zeros((iteration, 2), dtype=np.int)
    currx = current_pointsx[0]
    curry = current_pointsy[0]
    for i in range(iteration - 1, -1, -1):  # iterate through path backwards, using directionmap to know which shift
        path[i, :] = currx, curry
        currx -= shifts[direction_map[currx, curry, :]][0, 0]
        curry -= shifts[direction_map[path[i, 0], curry, :]][0, 1]
        # path[i,0] instead of currx since currx changes and to avoid temp var

    return path


if __name__ == "__main__":
    import time
    from matplotlib import pyplot as plt

    # Open skeletons and seed points
    skeletons = np.load('C:\\Users\\mearns\\Documents\Python\\walk_skeleton_for_joe\\skeletons.npy')
    seeds = np.load('C:\\Users\\mearns\\Documents\Python\\walk_skeleton_for_joe\\seeds.npy')

    idx = 884  # branchiest
    # idx = 410  # least branchy

    print('starting')
    niter = 10
    t0 = time.time()
    for j in range(niter):
        paths = walk_skeleton(skeletons[idx], seeds[idx], first_walk=True)
        longest_path = paths[np.argmax([len(p) for p in paths])]
    print('original time', (time.time() - t0) / niter)

    t1 = time.time()
    for j in range(niter):
        # fast_longestpath(skeletons[idx][min0:max0, min1:max1], seeds[idx][::-1] - [min0, min1])
        longest_path_new = fast_longestpath(skeletons[idx], seeds[idx][::-1])
    print('new time', (time.time() - t1) / niter)

    # They match, though I don't include the seed point and it can be off by a point or two at the start
    # assert (np.isclose(longest_path[2:, :], longest_path_new)).all()
    plt.imshow(skeletons[idx])
    plt.scatter(longest_path_new[:, 1] + .1, longest_path_new[:, 0], s=20, c='r', alpha=.5)
    plt.scatter(longest_path[:, 1], longest_path[:, 0] + .1, s=20, c='b', alpha=.5)
    plt.scatter(*seeds[idx].astype(np.int), c='pink')
    plt.show()
