import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os
import imageio
from scipy.stats import multivariate_normal
import argparse
import csv


def plot1d(data, K, pi, mu, sigma, iter):
    """
    Function for plotting 1D data
    --------------------
    Parameters  :
    data        : Original data array to plot
    K           : Number of clusters
    pi          : Probability of selecting k th gaussian
    mu          : Mean of each gaussian distribution
    sigma       : Variance of each gaussian distribution
    iter        : Current iteration
    """
    x = np.linspace(np.min(data), np.max(data), 100)
    y = np.zeros(100)
    for i in range(K):
        y += pi[i] * multivariate_normal.pdf(x, mu[i], sigma[i])

    plt.scatter(data, np.zeros_like(data), s=2, label='original data')
    plt.plot(x, y, c='orange', label='GMM')
    plt.scatter(mu, np.zeros_like(mu), marker='x', c='r', label='centroids')

    plt.xlabel('x')
    plt.ylabel('probability density')
    plt.legend(loc='upper right')
    plt.title(f'1D GMM (K = {K}), Iter = {iter}')
    plt.savefig(f"result_iter={iter}.png")
    plt.clf()


def plot2d_3D(data, K, pi, mu, sigma, iter, fig):
    """
    Function for plotting 2D data
    --------------------
    Parameters  :
    data        : Original data array to plot
    K           : Number of clusters
    pi          : Probability of selecting k th gaussian
    mu          : Mean of each gaussian distribution
    sigma       : Variance of each gaussian distribution
    iter        : Current iteration
    """
    data_x, data_y = data.T
    mu_x, mu_y = mu.T

    x = np.linspace(np.min(data_x), np.max(data_x), 100)
    y = np.linspace(np.min(data_y), np.max(data_y), 100)
    X, Y = np.meshgrid(x, y)
    XY = np.dstack([X, Y])
    Z = np.zeros((100, 100))
    for i in range(K):
        Z += pi[i] * multivariate_normal.pdf(XY, mu[i], sigma[i])

    ax = Axes3D(fig)
    ax.set_title(f"2D GMM (K = {K}), iter = {iter}", size=20)
    ax.scatter3D(data_x, data_y, np.zeros_like(
        data_x), s=2, label='original data')
    ax.contour3D(X, Y, Z, 50)
    ax.scatter(mu_x, mu_y, np.zeros_like(mu_x), s=40, linewidths=3,
               c='r', marker='x', label='Calculated centroids')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('probability density')

    ax.legend()
    ax.view_init(azim=70, elev=40)
    fig.savefig(f"result_iter={iter}.png")
    plt.clf()


def plot_likelihood(likelihood, filename):
    """
    Function for plotting log likelihood
    """
    plt.plot(likelihood)
    plt.title(f'log likelihood of {filename}')
    plt.xlabel('iteration')
    plt.ylabel('log likelihood')
    plt.savefig('log_lokelihood.png')
    plt.clf()


def rotate_3D():
    """
    Function for ganerating rotating 3D plot
    """
    fig = plt.figure()
    ax = Axes3D(fig)

    def rotate(angle):
        ax.view_init(azim=angle, elev=40)
    rot_animation = animation.FuncAnimation(
        fig, rotate, frames=100, interval=50)
    rot_animation.save('rotation3D.gif', dpi=80)


def make_gif(iter):
    """
    Function for combining every .png file into .gif file and erase .png files
    """
    filenames = []
    for i in range(iter):
        filename = f'result_iter={i}.png'
        filenames.append(filename)

    # build gif
    with imageio.get_writer('moving_iter.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files
    for filename in set(filenames):
        os.remove(filename)

"""
def gauss(x, mean, cov):
    dim = x.shape[0]
    cov_inv = np.linalg.inv(cov)
    coef = 1 / ((2*np.pi) ** (dim / 2) * np.sqrt(np.linalg.det(cov)))
    gaussian = coef * np.exp(-((x-mean).T @ cov_inv @ (x-mean))/2)
    return gaussian
"""

def em_algo(x, K, D, mu, sigma, pi):
    q = np.zeros([x.shape[0], K])
    # E step
    for j in range(K):
        dist = x - mu[j]
        inv_sigma = np.linalg.inv(sigma[j])
        mahalanobis = np.sum(dist @ inv_sigma * dist, axis=1)
        exp = np.exp(-1 * mahalanobis/2)
        det_sigma = np.linalg.det(sigma[j])
        q[:, j] = pi[j] * exp / np.sqrt(det_sigma)
    gamma = q / np.sum(q, axis=1, keepdims=True)

    # M step
    mu = (gamma.T @ x) / np.sum(gamma, axis=0, keepdims=True).T
    for k in range(K):
        dist = x - mu[k]
        sigma[k] = ((np.tile(gamma[:, k], (x.shape[1], 1))
                     * dist.T) @ dist) / np.sum(gamma[:, k])
    pi = np.sum(gamma, axis=0) / np.sum(gamma)

    return pi, mu, sigma, q


def gmm_classification(x, K, D, max_iter):
    # Initialize each variable
    N = len(x)
    mu = np.linspace(np.min(x), np.max(x), K)
    sigma = np.zeros((K, D, D))
    matrix = np.zeros((D, D))
    for i in range(K):
        for j in range(D):
            matrix[j][j] = np.random.uniform(1, 2)
            matrix[D-j-1][j] = np.random.uniform(0, 1)
        sigma[i] = matrix
    pi = [1/K] * K

    old_likelihood = 0
    new_likelihood = 1
    likelihood = []
    fig = plt.figure()
    for i in range(max_iter):
        if abs(old_likelihood - new_likelihood) < 0.01:
            break
        old_likelihood = new_likelihood
        pi, mu, sigma, q = em_algo(x, K, D, mu, sigma, pi)
        new_likelihood = np.sum(np.log(np.sum(q/(2*np.pi) ** (D/2), axis=1)))
        likelihood.append(new_likelihood)
        if D == 1:
            plot1d(x, K, pi, mu, sigma, i)
        elif D == 2:
            plot2d_3D(x, K, pi, mu, sigma, i, fig)

    make_gif(i)

    return pi, mu, sigma, likelihood


def main():
    parser = argparse.ArgumentParser(
        description='Program for fitting GMM model')
    parser.add_argument("-f", dest="filename", help='Filename', required=True)
    parser.add_argument("-k", dest="K", type=int,
                        help='Number of clusters', required=False, default=3)
    parser.add_argument("-i", dest="max_iter", type=int,
                        help='Number of clusters', required=False, default=50)
    args = parser.parse_args()
    data = pd.read_csv(args.filename, header=None).values
    dim = np.shape(data)[1]
    K = args.K
    max_iter = args.max_iter
    pi, mu, sigma, likelihood = gmm_classification(data, K, dim, max_iter)
    plot_likelihood(likelihood, args.filename)


if __name__ == "__main__":
    main()
