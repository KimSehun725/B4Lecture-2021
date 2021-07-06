import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt

import ex_5.a_miyashita.kmeans as km


def gaussian(x, mu, sigma):
    """
        Calculate plausibility of gaussian distribution.
        # Args
            x (ndarray, axis=(sample, dim)): input data
            mu (ndarray, axis=(k, dim)): mean vector
            sigmma (ndarray, axis=(k, dim, dim)): covariance matrix
        # Returns
            out (ndarray, axis=(samples, k)): plausibility
    """
    # (samplesize, k, dim)<-(samplesize, 1, dim)-(k, dim)
    dev = x[:,np.newaxis,:] - mu
    # (k,)<-(k, dim, dim)
    det = np.linalg.det(sigma)
    # (k, dim, dim)<-(k, dim, dim)
    lamda = np.linalg.inv(sigma)
    # (samplesize,)<-(samplesize, k, row),(k, row, col),(samplesize, k, col)
    z = np.einsum('skr,krc,skc->sk', dev, lamda, dev)
    # (samplesize, k)<-(samplesize, k)/(k,)
    return np.exp(-0.5*z)/np.sqrt(2*np.pi*det)


class Gmm:
    def __init__(self, k, dim):
        """
            Class of mixtures of gaussian.
            # Args
                k (int): number of components of GMM
                dim (int): dimension of data
        """
        self.k = k
        self.dim = dim
        # mixture coefficient
        self.pi = np.zeros(k)
        # mean
        self.mu = np.zeros((k, dim))
        # covariance matrix
        self.sigma = np.zeros((k, dim, dim))

    def fit(self, x, y=None):
        """
            Fit parameters by EM.
            # Args
                x (ndarray, axis=(samples, dim)): input data
                y (None): dummy
        """
        if x.shape[1] != self.dim:
            print(f"GMM.fit: x has invalid dimension {x.shape[1]}")
            sys.exit()
        
        samplesize = x.shape[0]

        # initialize parameters
        means_init = km.k_meanstt(x, self.k)
        means, labels = km.k_means(x, means_init, one_hot=True)
        # (k,)<-(samplesize, k)
        clustersize = np.sum(labels, axis=0)
        # (k,)<-(k,)/()
        self.pi = clustersize/np.sum(clustersize)
        # (k, dim)
        self.mu = means
        # (samplesize, k, dim)<-(samplesize, 1, dim)-(k, dim)
        dev = x[:,np.newaxis,:] - self.mu
        # (k, row, col)<-(samplesize, k),(samplesize, k, row),(samplesize, k, col)
        self.sigma = np.einsum('sk,skr,skc->krc', labels, dev, dev)
        # (k, row, col)<-(k, row, col)/(k, 1, 1)
        self.sigma = self.sigma/clustersize[:,np.newaxis,np.newaxis]

        prev = -np.inf
        plauss = np.array([])

        while 1:
            # calculate plausibility and responsibility
            # (samplesize, k)<-(k,)*(samplesize, k)
            gamma = self.pi*gaussian(x, self.mu, self.sigma)
            plaus = np.sum(gamma, axis=1)
            gamma = gamma/plaus[:,np.newaxis]
            plaus = np.sum(np.log(plaus))
            if np.abs(plaus - prev) < 1e-16:
                break
            prev = plaus
            plauss = np.append(plauss, plaus)

            # optimize parameters
            # (k,)<-(samplesize, k)
            n = np.sum(gamma, axis=0)
            # (k, dim)<-(k, samplesize)@(samplesize, dim)
            self.mu = gamma.T @ x
            # (k, dim)<-(k, dim)/(k, 1)
            self.mu = self.mu/n[:,np.newaxis]
            # (samplesize, k, dim)<-(samplesize, 1, dim)-(k, dim)
            dev = x[:,np.newaxis,:] - self.mu
            # (k, row, col)<-(samplesize, k),(samplesize, k, row),(samplesize, k, col)
            self.sigma = np.einsum('sk,skr,skc->krc', gamma, dev, dev)
            # (k, row, col)<-(k, row, col)/(k, 1, 1)
            self.sigma = self.sigma/n[:,np.newaxis,np.newaxis]
            # (k,)<-(k,)/()
            self.pi = n/samplesize
        
        plt.plot(plauss)
        plt.xlabel("iteration")
        plt.ylabel("Log plausibility")
        plt.show()

    def predict(self, x):
        """
            Calculate plausibility of given data according to learnt parameters.
            # Args
                x (ndarray, axis=(samples, dim)): input data

            # Returns
                p (ndarray, axis=(samples,)): plausibility
        """
        samplesize = x.shape[0]
        # (samplesize, k)<-(k,)*(samplesize, k)
        p = self.pi*gaussian(x, self.mu, self.sigma)
        p = np.sum(p, axis=1)
        return p


def main():
    # process args
    parser = argparse.ArgumentParser(description="GMM fitting")
    parser.add_argument("sc", type=str, help="input filename followed by extention .csv")
    parser.add_argument("k", type=int, help="number of components of model")
    args = parser.parse_args()

    data = np.loadtxt(f"{args.sc}.csv", delimiter=',')
    print(f">>> data = np.loadtxt({args.sc}.csv, delimiter=',')")
    print(">>> print(data.shape)")
    print(data.shape)

    if data.ndim == 1:
        data = data[:,np.newaxis]

    if data.shape[1] == 1:
        # plot
        plt.hist(data[:,0])
        plt.title(args.sc)
        plt.xlabel("$x_0$")
        plt.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=1)
        plt.show()

        # model fitting and prediction
        model = Gmm(args.k, 1)
        model.fit(data)
        x = np.linspace(data.min(), data.max(), 500)
        p = model.predict(x[:,np.newaxis])

        #plot
        plt.plot(x, p, label="predicted distribution")
        plt.scatter(data[:,0], np.zeros_like(data[:,0]), facecolor='None', edgecolors='g', label="observed data")
        plt.scatter(model.mu[:,0], np.zeros_like(model.mu[:,0]), marker='x', color='r', label="centroids")
        plt.title(args.sc)
        plt.xlabel("$x$")
        plt.ylabel("$p(x)$")
        plt.legend()
        plt.show()

    if data.shape[1] == 2:
        # plot
        plt.scatter(data[:,0], data[:,1], facecolor='None', edgecolors='r')
        plt.title(args.sc)
        plt.xlabel("$x_0$")
        plt.ylabel("$x_1$")
        plt.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=1)
        plt.show()

        # model fitting and prediction
        model = Gmm(args.k, 2)
        model.fit(data)

        left = data[:,0].min()
        right = data[:,0].max()
        d = (right-left)*0.05
        x = np.linspace(left-d, right+d, 500)

        left = data[:,1].min()
        right = data[:,1].max()
        d = (right-left)*0.05
        y = np.linspace(left-d, right+d, 500)

        xx, yy = np.meshgrid(x, y)
        xy = np.array([xx.ravel(), yy.ravel()]).T

        p = model.predict(xy)
        p = p.reshape(500, 500)

        plt.contour(xx, yy, p, levels=15)
        plt.scatter(data[:,0], data[:,1], facecolor='None', edgecolors='r', label="observed data")
        plt.scatter(model.mu[:,0], model.mu[:,1], marker='x', label="centroids")
        plt.title(args.sc)
        plt.xlabel("$x_0$")
        plt.ylabel("$x_1$")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()    