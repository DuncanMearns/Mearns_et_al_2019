from scipy.stats import mannwhitneyu as scipy_mannshitneyu
import numpy as np
from itertools import permutations


def u_statistic(ranks):
    ranks = np.array(ranks)
    ranks_1 = np.where(ranks == 0)[0] + 1
    ranks_2 = np.where(ranks == 1)[0] + 1
    U1 = np.sum(ranks_1) - (0.5 * len(ranks_1) * (len(ranks_1) + 1))
    U2 = np.sum(ranks_2) - (0.5 * len(ranks_2) * (len(ranks_2) + 1))
    return min(U1, U2)


def mannwhitneyu(x, y, alternative='two-sided'):
    nx, ny = len(x), len(y)
    if (ny <= 20) or (ny <= 20):
        all_values = np.concatenate([x, y])
        all_labels = np.concatenate([np.zeros((nx,)), np.ones((ny,))]).astype('uint8')
        ranked = all_labels[np.argsort(all_values)]
        if len(ranked) <= 10:  # compute exactly
            null_distribution = np.array([u_statistic(ranks) for ranks in permutations(ranked)])
        else:
            null_distribution = np.array([u_statistic(np.random.permutation(ranked)) for iteration in range(1000)])
        # return null_distribution
        u = u_statistic(ranked)
        p_value_less = (null_distribution < u).sum() / float(len(null_distribution))
        p_value_more = 1 - (null_distribution > u).sum() / float(len(null_distribution))
        if alternative == 'less':
            if np.median(x) < np.median(y):
                p_value = p_value_less / 2.
            else:
                p_value = 1 - (p_value_less / 2.)
        elif alternative == 'greater':
            if np.median(x) > np.median(y):
                p_value = p_value_more / 2.
            else:
                p_value = 1 - p_value_more / 2.
        elif alternative == 'two-sided':
            p_value = min(p_value_less, p_value_more)
        else:
            raise ValueError('wrong argument passed to alternative')
    else:
        u, p_value = scipy_mannshitneyu(x, y, alternative=alternative)
    return u, p_value


if __name__ == "__main__":

    from scipy.stats import norm

    x = norm(loc=0, scale=0.5).rvs(20)
    y = norm(loc=0.1, scale=0.5).rvs(20)

    print mannwhitneyu(x, y, 'less'), scipy_mannshitneyu(x, y, alternative='less')
