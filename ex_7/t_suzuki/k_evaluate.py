import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gauss(data, mean, sigma):
    """
    input
    data  : ndarray (n, dim)
            input data
    mean  : ndarray (k, dim)
            data means (centroid)
    sigma : ndarray (k, dim, dim)
            convariance matrix

    return
    gauss : ndarray
            Gaussian distribution
    """

    n = data.shape[0]
    dim = data.shape[1]
    gauss = np.array([])

    inv = np.linalg.inv(sigma)
    det = np.linalg.det(sigma)
    den = np.sqrt((2 * np.pi ** dim) * det)

    for i in range(n):
        num = np.exp(-0.5 * (data[i] - mean).T @ inv @ (data[i] - mean))
        gauss = np.append(gauss, num / den)

    return gauss


def gmm(data, mean, sigma, pi):
    """
    input
    data  : ndarray (n, dim)
            input data
    mean  : ndarray (k, dim)
            data means
    sigma : ndarray (k, dim, dim)
            convariance matrix

    return
    output : ndarray (k, n)
             pi * gaussian distribution
    prob   : ndarray ( , n)
             probability density
    """

    cls_num = len(mean)
    output = np.array([pi[k] * gauss(data, mean[k], sigma[k]) for k in range(cls_num)])
    return output, np.sum(output, axis=0)[np.newaxis, :]


def log_likelihood(data, mean, sigma, pi):
    """
    input
    data  : ndarray (n, dim)
            input data
    mean  : ndarray (k, dim)
            data means
    sigma : ndarray (k, dim, dim)
            convariance matrix
    pi    : ndarray (k)
            mixing coefficient

    return
    likelihood : float
                 log likelihood function
    """

    _, gmm_sum = gmm(data, mean, sigma, pi)
    likelihood = np.sum(np.log(gmm_sum), axis=1)
    return likelihood


def em_algorithm(data, cls_num, mean, sigma, pi, thr):
    """
    input
    data    : ndarray (n, dim)
              input data
    cls_num : int
              number of cluster
    mean    : ndarray (k, dim)
              data means
    sigma   : ndarray (k, dim, dim)
              convariance matrix
    pi      : ndarray (k)
              mixing coefficient
    thr     : float
              threshold of EM algorithm

    return
    log_list : list of float
               log likelihood values
    mean     : ndarray (k, dim)
               data means
    sigma    : ndarray (k, dim, dim)
               convariance matrix
    pi       : ndarray (k)
               mixing coefficient
    """

    cnt = 0
    n, dim = data.shape
    log_list = np.array(log_likelihood(data, mean, sigma, pi))

    while True:
        # E-step
        gmm_com, gmm_sum = gmm(data, mean, sigma, pi)
        gamma = gmm_com / gmm_sum

        # M-step
        Nk = np.sum(gamma, axis=1)
        mean = (gamma @ data) / Nk[:, np.newaxis]
        sigma_list = np.zeros((n, cls_num, dim, dim))
        for k in range(cls_num):
            for i in range(n):
                sigma_com = gamma[k][i] * (data[i] - mean[k])[:, np.newaxis] @ ((data[i] - mean[k]))[np.newaxis, :]
                sigma_list[i][k] = sigma_com
        sigma = np.sum(sigma_list, 0) / Nk[:, np.newaxis, np.newaxis]
        pi = Nk / n

        # update log-likelihood
        log_list = np.append(log_list, log_likelihood(data, mean, sigma, pi))

        if np.abs(log_list[cnt] - log_list[cnt+1]) < thr:
            break
        else:
            cnt += 1

    return log_list, mean, sigma, pi


def aic(likelihood, cls_num, dim):
    """
    input
    likelihood : float
                 last likelihood function value
    cls_num    : int
                 number of cluster
    n          : int
                 number of data
    dim        : int
                 data dimensions

    return
    aic : float
          aic value
    """

    aic = -2 * likelihood + 2 * cls_num * (1 + dim + dim ** 2)
    return aic


def bic(likelihood, cls_num, n, dim):
    """
    input
    likelihood : float
                 last likelihood function value
    cls_num    : int
                 number of cluster
    n          : int
                 number of data
    dim        : int
                 data dimensions

    return
    aic : float
          aic value
    """

    bic = -2 * likelihood + cls_num * (1 + dim + dim ** 2) * np.log(n)
    return bic


def main():
    # argments settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input data file path")
    parser.add_argument("--lim", type=int, default=8, help="number of cluster limit")
    parser.add_argument("--thr", type=float, default=0.001, help="threshold value of EM algorithm")
    args = parser.parse_args()

    lim = args.lim + 1
    thr = args.thr

    # check output dir exist
    if not os.path.exists('./out'):
        os.makedirs('./out')

    # loading data
    data = pd.read_csv(args.input, header=None).to_numpy()
    n, dim = data.shape

    aic_list = []
    bic_list = []

    print(f'Calculating AIC and BIC from k = 1 to {lim-1} ...')
    for cls_num in range(1, lim):
        
        # parameter initialization
        pi = np.zeros(cls_num)
        sigma = np.zeros((cls_num, dim, dim))
        mean = np.zeros((cls_num, dim))
        for i in range(cls_num):
            pi[i] = 1 / cls_num
            sigma[i] = np.cov(data.T, bias=True)
            mean[i] = (np.mean(data[n*i//cls_num:n*(i+1)//cls_num], axis=0))

        # EM Algorithm
        likelihood, mean, sigma, pi = em_algorithm(data, cls_num, mean, sigma, pi, thr)
        print(f'Finish: k = {cls_num}')
        aic_list.append(aic(likelihood[-1], cls_num, dim))
        bic_list.append(bic(likelihood[-1], cls_num, n, dim))
        
    # plot bic
    plt.plot([x for x in range(1, lim)], aic_list, label='AIC')
    plt.plot([x for x in range(1, lim)], bic_list, label='BIC')
    plt.xlabel('Cluster num')
    plt.ylabel('Imformation criterion')
    plt.legend()
    plt.show(block=True)
    plt.close()

    print(f'AIC Best Cluster num is {aic_list.index(min(aic_list)) + 1}')
    print(f'BIC Best Cluster num is {bic_list.index(min(bic_list)) + 1}')


if __name__ == '__main__':
    main()
