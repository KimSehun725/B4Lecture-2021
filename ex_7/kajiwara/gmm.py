import numpy as np


def gaussian(x, mu, sigma):
    """
    calculate multi dimention gauss distribution

    Parameters
    ----------
    x: ndarray (dim, )
        one sample of data
    mu: ndarray (dim, )
        means of distribution
    sigma: ndarray (dim, )
        variances of distribution

    Returns
    -------
    gaussian: ndarray (dim, )
        gauss distribution
    """

    dim = len(x)

    sigma_inv = np.linalg.inv(sigma)
    exp = (x - mu).T @ sigma_inv @ (x - mu)
    exp /= 2

    denomination = (2*np.pi)**dim * np.linalg.det(sigma)

    gaussian = np.exp(-exp) / np.sqrt(denomination)

    return gaussian


def calc_gaussian(X, mu, sigma):
    """
    caluculate gaussian of each sample of input data

    Parameters
    ----------
    X: ndarray (n, dim)
        input data
    mu: ndarray (dim, )
        means of distribution
    sigma: ndarray (dim, dim)
        variances of distribution (dim, )

    Returns
    -------
    gaussians: ndarray (n, )
        gaussian list of each sample in input
    """

    n = len(X)
    gaussians = np.zeros(n)
    for i, x in enumerate(X):
        gaussians[i] = gaussian(x, mu, sigma)

    return gaussians


def mixture_gaussian(X, Pi, Mu, Sigma):
    """
    caluculate mixture gauss distribution

    Parameters
    ----------
    X: ndarray (n, dim)
        input data
    Pi: (class, dim)
        mixing coefficient of each class
    Mu: ndarray (class num, dim)
        means of each class
    Sigma: ndarray (class, dim, dim)
        variances of each class

    Returns
    -------
    m_gaussian: ndarray (class, n)
        mixture gauss distribution
    """

    n = len(X)
    k = len(Pi)
    m_gaussian = np.zeros((k, n))
    for j in range(k):
        m_gaussian[j] = Pi[j]*calc_gaussian(X, Mu[j], Sigma[j])

    return m_gaussian


def log_likelihood(X, Pi, Mu, Sigma):
    """
    calc log likelihood

    Parameters
    ----------
    X: ndarray (n, dim)
        input data
    Pi: (class, dim)
        mixing coefficient of each class
    Mu: ndarray (class num, dim)
        means of each class
    Sigma: ndarray (class, dim, dim)
        variances of each class

    Returns
    -------
    log_l: float
        log likelihood
    """

    n = len(X)

    m_gaussian = mixture_gaussian(X, Pi, Mu, Sigma)
    sum_g = np.sum(m_gaussian, axis=0)

    log_l = 0
    for i in range(n):
        log_l += np.log(sum_g[i])

    return log_l


def em_algorithm(X, Pi, Mu, Sigma, epsilon, output=True):
    """
    run EM algorithm

    Parameters
    ----------
    X: ndarray (n, dim)
        input data
    Pi: (class, dim)
        mixing coefficient of each class
    Mu: ndarray (class num, dim)
        means of each class
    Sigma: ndarray (class, dim, dim)
        variances of each class

    Returns
    -------
    ite: int 
        count of iteration
    likelihoods: list[float]
        likelihood list of each iteration
    Pi: (class, dim)
        mixing coefficient of each class
    Mu: ndarray (class num, dim)
        means of each class
    Sigma: ndarray (class, dim, dim)
        variances of each class
    """

    if output:
        print('init parameters')
        print(f'means:\n {Mu}\n variances:\n {Sigma}\n mixture coef:\n {Pi}\n')

    ite = 0
    likelihoods = []
    means = []
    covs = []
    log_l = log_likelihood(X, Pi, Mu, Sigma)

    likelihoods.append(log_l)
    means.append(Mu.copy())
    covs.append(Sigma.copy())
    while True:
        # E step
        N = len(X)

        m_gaussian = mixture_gaussian(X, Pi, Mu, Sigma)
        burden_rate = m_gaussian / np.sum(m_gaussian, axis=0)[None:, ]

        # M step
        N = len(X)
        K = len(Pi)
        N_k = np.sum(burden_rate, axis=1)[:, None]

        Pi = N_k/N

        for k in range(K):
            Sigma[k] = 0
            for n in range(N):
                Sigma[k] += burden_rate[k][n]*(X[n] - Mu[k])[:, None]@((X[n] - Mu[k]))[None, :]
        Sigma = Sigma / N_k[:, None]

        Mu = (burden_rate @ X) / N_k

        l_new = log_likelihood(X, Pi, Mu, Sigma)
        gap = l_new-likelihoods[ite]

        if output:
            print(f'iteration: {ite},  likelihood: {l_new},  gap: {gap}')

        if gap < epsilon:
            if output:
                print('='*10)
                print(f'finished  iteration: {ite},  likelihood: {l_new}')
                print(f'means:\n {Mu}\n variances:\n {Sigma}\n mixture coef:\n {Pi}\n')
                print('='*10)
            break

        ite += 1
        likelihoods.append(l_new)
        means.append(Mu.copy())
        covs.append(Sigma.copy())

    return ite, likelihoods, Mu, Sigma, Pi, means, covs
