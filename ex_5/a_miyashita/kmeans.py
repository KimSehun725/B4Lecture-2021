import argparse

import numpy as np
from matplotlib import pyplot as plt


def lbg(data, m, one_hot=False):
    """
        Clustering by LBG.

        # Args
            data (ndarray, shape=(sample_size, dim)): input data.
            m (int): 2**m is size of codebook.
            one_hot (bool): 
                If one_hot is True, one-hot-vector is returned instead of cluster label.

        # Returns
            means (ndrray, shape=(cluster_size, dim)):
                mean of each cluster (codebook).
            labels (ndarray, shape={(sample_size), (sample_size, cluster_size)}):
                label of cluster assigned to each data. If one_hot is True, 
                one-hot-vector correspond to cluster label.
    """
    means = np.mean(data, axis=0).reshape(1, -1)
    for i in range(m):
        e = np.random.rand(data.shape[1])*2-1
        means = np.concatenate([means+e, means-e], axis=0)
        means, labels = k_means(data, means, one_hot)
    return means, labels


def k_meanstt(data, n):
    """
        Generate initial value for k-means by k-means++.

        # Args
            data (ndarray, shape=(sample_size, dim)): input data.
            n (int): size of cluster.

        # Returns
            means_init (ndrray, shape=(cluster_size, dim)): initial value of means
    """
    rng = np.random.default_rng()
    c = rng.choice(data)
    means_init = np.zeros((n, data.shape[1]))
    means_init[0] = c
    D = np.sum((data-c)**2, axis=1)
    for i in range(1, n):
        p = D/np.sum(D)
        c = rng.choice(data, p=p)
        means_init[i] = c
        d = np.sum((data-c)**2, axis=1)
        D = np.minimum(D, d)
    return means_init


def minimax(data, n):
    """
        Generate initial value for k-means by minimax.

        # Args
            data (ndarray, shape=(sample_size, dim)): input data.
            n (int): size of cluster.

        # Returns
            means_init (ndrray, shape=(cluster_size, dim)): initial value of means
    """
    rng = np.random.default_rng()
    c = rng.choice(data)
    means_init = np.zeros((n, data.shape[1]))
    means_init[0] = c
    d = np.linalg.norm(data-c, axis=1)
    for i in range(1, n):
        c = data[np.argmax(d)]
        means_init[i] = c
        d += np.linalg.norm(data-c, axis=1)
    return means_init


def k_means(data, means_init, one_hot=False):
    """
        Clustering by k-means.

        # Args
            data (ndarray, shape=(sample_size, dim)): input data.
            means_init (ndrray, shape=(cluster_size, dim)): initial value of means
            one_hot (bool): 
                If one_hot is True, one-hot-vector is returned instead of cluster label.

        # Returns
            means (ndrray, shape=(cluster_size, dim)):
                mean of each cluster (codebook).
            labels (ndarray, shape={(sample_size), (sample_size, cluster_size)}):
                label of cluster assigned to each data. If one_hot is True, 
                one-hot-vector correspond to cluster label.
    """
    means = means_init
    while(1):
        d = np.linalg.norm(
                data.reshape(data.shape[0], 1, -1)-means, axis=-1
            )
        labels = np.nanargmin(d, axis=1)
        labels_oh = np.eye(means_init.shape[0])[labels]
        g = labels_oh.T @ data / np.sum(labels_oh, axis=0).reshape(-1, 1)
        if np.all(means[~np.isnan(g)] == g[~np.isnan(g)]):
            break
        means = g

    if one_hot:
        labels = labels_oh
    
    return means, labels


def main():
    # process args
    parser = argparse.ArgumentParser(description="Split datas into cluster")
    parser.add_argument("sc", type=str, help="input filename with extension .csv")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lbg", type=int, help="size of cluster for lbg")
    group.add_argument("--meanstt", type=int, help="size of cluster for k-means++")
    group.add_argument("--minimax", type=int, help="size of cluster for minimax")
    args = parser.parse_args()

    data = np.loadtxt(args.sc, delimiter=',', skiprows=1)
    
    if data.ndim == 1:
        data = data[:,np.newaxis]
        data = np.concatenate([data, np.zeros_like(data)],axis=1)

    if args.lbg:
        means, labels = lbg(data, args.lbg)

    else:
        if args.meanstt:
            means_init = k_meanstt(data, args.meanstt)

        elif args.minimax:
            means_init = minimax(data, args.minimax)

        means, labels = k_means(data, means_init)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax = fig.add_subplot(111, projection="3d")

    for i in range(means.shape[0]):
        plt.scatter(data[:, 0][labels == i], data[:, 1][labels == i])
        # ax.scatter3D(data[:, 0][labels == i], data[:, 1][labels == i], data[:, 2][labels == i])

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    # ax.set_zlabel('$x_3$')
    plt.show()

if __name__ == "__main__":
    main()
