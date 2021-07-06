import numpy as np

import utils


class KMeans():
    def __init__(self, n_clusters=3, init='random', max_ite=300, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_ite = max_ite
        self.random_state = random_state

    def init_centroids(self, X, init, random_state):
        n_samples = X.shape[0]
        n_clusters = self.n_clusters

        if init == 'random':
            seeds = random_state.permutation(n_samples)[:n_clusters]
            centroids = X[seeds]
        else:
            pass

        return centroids

    def fit(self, X):
        n_samples = X.shape[0]
        random_state = utils.check_random_state(self.random_state)
        centroids = self.init_centroids(X, 'random', random_state)
        new_centroids = centroids.copy()

        clusters = np.zeros(n_samples)

        for _ in range(self.max_ite):
            for i in range(n_samples):
                distances = np.sum((centroids - X[i])**2, axis=1)
                clusters[i] = np.argsort(distances)[0]

            for i in range(self.n_clusters):
                new_centroids[i] = X[clusters == i].mean(axis=0)

            if np.sum(new_centroids == centroids) == 4:
                break
            centroids = new_centroids

        self.centroids = centroids
        self.clusters = clusters

        return self

    def predict(self, X):
        n_samples = X.shape[0]
        pred = np.zeros(n_samples)

        for i in range(n_samples):
            distances = np.sum((self.centroids - X[i])**2, axis=1)
            pred[i] = np.argsort(distances)[0]

        return pred

    def fit_predict(self, X):
        return self.fit(X).clusters
