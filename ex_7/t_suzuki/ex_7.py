import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


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
    prob = np.sum(output, axis=0)[np.newaxis, :]
    return output, prob


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

    _, prob = gmm(data, mean, sigma, pi)
    likelihood = np.sum(np.log(prob), axis=1)
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


def main():
    # argments settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input data file path")
    parser.add_argument("--k", type=int, required=True, help="number of cluster")
    parser.add_argument("--thr", type=float, default=0.001, help="threshold value of EM algorithm")
    args = parser.parse_args()
    cls_num = args.k
    thr = args.thr

    # check output dir exist
    if not os.path.exists('./out'):
        os.makedirs('./out')

    # loading data
    data = pd.read_csv(args.input).to_numpy()
    data_name = args.input.split('/')[1].split('.')[0]
    n, dim = data.shape

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

    # 1d data GMM plot
    if dim == 1:
        x = np.linspace(np.min(data) - 1, np.max(data) + 1, 100)
        y = np.zeros(100)
        for k in range(cls_num):
            y += pi[k] * multivariate_normal.pdf(x, mean[k], sigma[k])

        fig = plt.figure()
        plt.scatter(data[:, 0], np.zeros(len(data[:, 0])), facecolor='None', edgecolors='skyblue', label='Data sample')
        plt.plot(x, y, c='orange', label='GMM')
        plt.scatter(mean, np.zeros(cls_num), marker='x', c='red', label='Centroids')
        plt.xlabel('X')
        plt.ylabel('Probability density')
        plt.title(f'K = {cls_num}')
        plt.grid()
        plt.legend()
        plt.show(block=True)
        fig.savefig(f'./out/gmm_{data_name}_{cls_num}.png')
        plt.close()

    # 2d data GMM plot
    else:
        x1 = np.linspace(np.min(data[:, 0]) - 1, np.max(data[:, 0]) + 1, 100)
        x2 = np.linspace(np.min(data[:, 1]) - 1, np.max(data[:, 1]) + 1, 100)
        x1, x2 = np.meshgrid(x1, x2)
        Z = np.dstack((x1, x2))

        prob = np.array([np.squeeze(gmm(z, mean, sigma, pi)[1], 0) for z in Z])
        fig = plt.figure()
        plt.scatter(data[:, 0], data[:, 1], facecolor='None', edgecolor='skyblue', label='Data sample')
        plt.scatter(mean[:, 0], mean[:, 1], marker='x', c='red', label='Centroids')
        cont = plt.contour(x1, x2, prob, cmap='rainbow')
        cont.clabel(fmt='%1.2f', fontsize=12)
        plt.xlabel('$X_{1}$')
        plt.ylabel('$X_{2}$')
        plt.title(f'K = {cls_num}')
        plt.grid()
        plt.legend()
        plt.show(block=True)
        fig.savefig(f'./out/gmm_{data_name}_{cls_num}.png')
        plt.close()
        
    # log likelihood
    fig = plt.figure()
    plt.plot(likelihood)
    plt.xlabel('Interation')
    plt.ylabel('Log Likelihood')
    plt.show(block=True)
    fig.savefig(f'./out/likelihood_{data_name}_{cls_num}.png')
    plt.close()


if __name__ == '__main__':
    main()
