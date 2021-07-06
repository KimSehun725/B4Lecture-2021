import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from gmm import em_algorithm
from utils import set_initial

TIME_TEMPLATE = '%Y%m%d%H%M%S'


def main(args):
    if args.result_path:
        result_path = Path(args.result_path)
        timestamp = datetime.now().strftime(TIME_TEMPLATE)
        result_path = result_path/timestamp
        if not result_path.exists():
            result_path.mkdir(parents=True)

    # loading data
    df1 = pd.read_csv('../data1.csv', header=None)
    df2 = pd.read_csv('../data2.csv', header=None)
    df3 = pd.read_csv('../data3.csv', header=None)

    # df to nd ndarray
    data1 = df1.values
    data2 = df2.values
    data3 = df3.values

    """data1"""
    k_1 = 2
    mu, sigma, pi = set_initial(data1, k_1)
    epsilon = 0.000001
    ite, likelihoods, mu, sigma, pi, _, _ = em_algorithm(data1, pi, mu, sigma, epsilon)

    x = np.linspace(np.min(data1), np.max(data1), 100)
    y = np.zeros(100)
    for i in range(k_1):
        y += pi[i] * st.multivariate_normal.pdf(x, mu[i], sigma[i])
    plt.scatter(data1, np.zeros_like(data1), s=2, label='original')
    plt.plot(x, y, label='GMM')
    plt.scatter(mu, np.zeros_like(mu), marker='x', label='centroid')
    plt.xlabel('x')
    plt.ylabel('probability density')
    plt.legend()
    plt.title('data1')
    plt.savefig(result_path / 'data1-1.png')
    plt.close()
    plt.clf()

    plt.plot([i for i in range(ite+1)], likelihoods)
    plt.xlabel('Iteration')
    plt.ylabel('log likelihood')
    plt.savefig(result_path / 'data1-2.png')
    plt.close()
    plt.clf()

    """data2"""
    k_2 = 3
    mu, sigma, pi = set_initial(data2, k_2)
    epsilon = 0.000001
    ite, likelihoods, mu, sigma, pi, _, _ = em_algorithm(data2, pi, mu, sigma, epsilon)

    n = len(data2)
    x = np.linspace(-1, 4, n)
    y = np.linspace(-1, 4, n)
    ax_x, ax_y = np.meshgrid(x, y)
    pos = np.dstack((ax_x, ax_y))
    z = np.zeros((n, n))
    for i in range(k_2):
        z += pi[i] * st.multivariate_normal.pdf(pos, mu[i], sigma[i])
    plt.contour(ax_x, ax_y, z)
    plt.scatter(data2[:, 0], data2[:, 1], facecolor='None', edgecolors='r')
    plt.scatter(mu[:, 0], mu[:, 1], marker='x')
    plt.title('data2')
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.savefig(result_path / 'data2-1.png')
    plt.close()
    plt.clf()

    plt.plot([i for i in range(ite+1)], likelihoods)
    plt.xlabel('Iteration')
    plt.ylabel('log likelihood')
    plt.savefig(result_path / 'data2-2.png')
    plt.close()
    plt.clf()

    """data3"""
    k_3 = 2
    mu, sigma, pi = set_initial(data3, k_3)
    epsilon = 0.000001
    ite, likelihoods, mu, sigma, pi, _, _ = em_algorithm(data3, pi, mu, sigma, epsilon)

    n = len(data3)
    x = np.linspace(-1, 4, n)
    y = np.linspace(-1, 4, n)
    ax_x, ax_y = np.meshgrid(x, y)
    pos = np.dstack((ax_x, ax_y))
    z = np.zeros((n, n))
    for i in range(k_3):
        z += pi[i] * st.multivariate_normal.pdf(pos, mu[i], sigma[i])
    plt.contour(ax_x, ax_y, z)
    plt.scatter(data3[:, 0], data3[:, 1], facecolor='None', edgecolors='r')
    plt.scatter(mu[:, 0], mu[:, 1], marker='x')
    plt.title('data3')
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.savefig(result_path / 'data3-1.png')
    plt.close()
    plt.clf()

    plt.plot([i for i in range(ite+1)], likelihoods)
    plt.xlabel('Iteration')
    plt.ylabel('log likelihood')
    plt.savefig(result_path / 'data3-2.png')
    plt.close()
    plt.clf()


if __name__ == "__main__":
    description = 'Example: python main.py ./result'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-rs', '--result-path', default='./result', help='path to save the result')

    args = parser.parse_args()

    main(args)
