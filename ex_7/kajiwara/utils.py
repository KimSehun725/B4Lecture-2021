import numbers

import numpy as np

from kmeans import KMeans


def check_random_state(seed):
    """
    Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def set_initial(X, k, mean_type='kmeans'):
    """
    initialize parameters of em algorithm

    Parameters
    ----------
    X: ndarray (n, dim)
        input data
    k: int
        number of class
    mean_type: 'kmeans' or 'random'
        method of initialization means
    Returns
    -------
    Pi: (class, dim)
        mixing coefficient of each class
    Mu: ndarray (class num, dim)
        means of each class
    Sigma: ndarray (class, dim, dim)
        variances of each class
    """

    n_samples = len(X)
    dim = X.shape[1]

    """means"""
    # mu = np.random.randn(k, dim)
    if mean_type == 'kmeans':
        resp = np.zeros((n_samples, k))
        km = KMeans(k)
        label = km.fit(X).clusters
        label = [int(i) for i in label]
        resp[np.arange(n_samples), label] = 1
    else:
        resp = np.random.mtrand._rand.rand(n_samples, k)
        resp /= resp.sum(axis=1)[:, np.newaxis]
    nk = resp.sum(axis=0) + 10*np.finfo(resp.dtype).eps
    mu = np.dot(resp.T, X) / nk[:, np.newaxis]

    """covariances"""
    # sigma = np.array([np.eye(dim) for _ in range(k)])
    sigma = np.empty((k, dim, dim))
    for kk in range(k):
        diff = X - mu[kk]
        sigma[kk] = np.dot(resp[:, kk] * diff.T, diff) / nk[kk]
        sigma[kk].flat[::dim + 1] += 1e-6

    """weight"""
    # pi = np.array([1/k for _ in range(k)])
    pi = nk / n_samples

    return mu, sigma, pi
