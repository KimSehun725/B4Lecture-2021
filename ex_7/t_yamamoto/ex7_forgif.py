# include flake8, black

import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def init(data, n_clusters):
    """
    Initialize each variable.

    Parameters:
        data : ndarray (N, D)
            Input data from csv file.
        n_clusters : int
            The number of clusters (n_clusters := K).

    Returns:
        pi : ndarray (K,)
            Initialized mixing coefficient.
        mu : ndarray (K, D)
            Initialized average value (standard normal distribution).
        sigma : ndarray (K, D, D)
            Initialized covariance matrix (identity matrix).
    """
    D = data.shape[1]
    pi = np.full(n_clusters, 1 / n_clusters)
    mu = np.random.randn(n_clusters, D)
    sigma = np.array([np.identity(D) for i in range(n_clusters)])
    return pi, mu, sigma


def gaussian(data, mu, sigma):
    """
    D-dimensional gaussian distribution.

    Initialize each variable.

    Parameters:
        data : ndarray (N, D)
            Input data from csv file.
        mu : ndarray (K, D)
            Average value.
        sigma : ndarray (K, D, D)
            Covariance matrix.

    Returns:
        ndarray (K, N)
            D-dimensional gaussian distribution.
    """
    D = data.shape[1]
    K = mu.shape[0]  # n_clusters
    diff_data = np.array([data - mu[i] for i in range(K)])  # (k, N, D)

    # mahalanobis distance
    mah_dist_sq = (
        diff_data @ np.linalg.inv(sigma) @ diff_data.transpose(0, 2, 1)
    )  # (k, N, N)
    mah_dist_sq = np.diagonal(mah_dist_sq, axis1=1, axis2=2)  # (k, N)

    # numerator
    nume = np.exp(-mah_dist_sq / 2)
    # denominator
    deno = np.sqrt(((2 * np.pi) ** D) * np.linalg.det(sigma)).reshape(-1, 1)

    return nume / deno


def gmm(data, pi, mu, sigma):
    """
    D-dimensional gaussian mixture model (GMM).

    Parameters:
        data : ndarray (N, D)
            Input data from csv file.
        pi : ndarray (K,)
            Mixing coefficient.
        mu : ndarray (K, D)
            Average value.
        sigma : ndarray (K, D, D)
            Covariance matrix.

    Returns:
        m_gauss : ndarray (N,)
            D-dimensional gaussian mixture model.
        w_gauss : ndarray (K, N)
            Weighted D-dimensional gaussian mixture model.
    """
    w_gauss = gaussian(data, mu, sigma) * pi.reshape(-1, 1)
    m_gauss = np.sum(w_gauss, axis=0)
    return m_gauss, w_gauss


def get_likelihood(m_gauss, log=True):
    """
    Calculate likelihood.

    Parameters:
        m_gauss : ndarray (N,)
            D-dimensional gaussian mixture model.
        log : bool, Default=True
            If log is True, log likelihood is calculated.

    Returns:
        likelihood : float
            Likelihood.
    """
    if log:
        likelihood = np.sum(np.log(m_gauss))
    else:
        likelihood = np.prod(m_gauss)
    return likelihood


def plot_1d(data, pi, mu, sigma, ftitle, iter=None):
    x = data[:, 0]
    K = mu.shape[0]  # n_clusters
    if iter is None:
        title = f"{ftitle} probability\nK = {K}"
    else:
        title = f"{ftitle} probability\nK = {K}, iter = {iter}"

    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title=title,
        xlabel="X",
        ylabel="Probability density",
    )
    ax.scatter(
        x,
        np.zeros(data.shape[0]),
        s=20,
        linewidths=1.0,
        marker="o",
        facecolor="None",
        edgecolors="darkblue",
        label="data",
    )
    ax.scatter(
        mu,
        np.zeros(mu.shape[0]),
        s=100,
        linewidths=3,
        marker="x",
        c="r",
        label="centroids",
    )
    x_scale = np.linspace(np.min(data), np.max(data), 100).reshape(100, 1)
    m_gauss, _ = gmm(x_scale, pi, mu, sigma)
    ax.plot(x_scale, m_gauss, label="GMM")
    # plt.tight_layout()
    plt.grid(ls=":")
    plt.legend()

    return fig, ax


def plot_2d(data, pi, mu, sigma, ftitle, iter=None):
    x = data[:, 0]
    y = data[:, 1]
    K = mu.shape[0]  # n_clusters
    X = np.linspace(np.min(x), np.max(x), 100)
    Y = np.linspace(np.min(y), np.max(y), 100)
    X, Y = np.meshgrid(X, Y)
    lines = np.dstack((X, Y))
    probability = np.array([gmm(line, pi, mu, sigma)[0] for line in lines])

    # 2D plot
    if iter is None:
        title = f"{ftitle} 2D-probability\nK = {K}"
    else:
        title = f"{ftitle} 2D-probability\nK = {K}, iter = {iter}"

    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        title=title,
        xlabel="X",
        ylabel="Y",
    )
    ax.scatter(
        x,
        y,
        s=20,
        linewidths=1.0,
        marker="o",
        facecolor="None",
        edgecolors="darkblue",
        label="data",
    )
    ax.scatter(
        mu[:, 0],
        mu[:, 1],
        s=100,
        linewidths=3,
        marker="x",
        c="r",
        label="centroids",
    )
    cset = ax.contour(X, Y, probability, cmap="rainbow")
    ax.clabel(cset, fontsize=9, inline=True)
    # plt.tight_layout()
    plt.grid(ls=":")
    plt.legend()

    return fig, ax


def plot_3d(data, pi, mu, sigma, ftitle, iter=None):
    x = data[:, 0]
    y = data[:, 1]
    K = mu.shape[0]  # n_clusters
    X = np.linspace(np.min(x), np.max(x), 100)
    Y = np.linspace(np.min(y), np.max(y), 100)
    X, Y = np.meshgrid(X, Y)
    lines = np.dstack((X, Y))
    probability = np.array([gmm(line, pi, mu, sigma)[0] for line in lines])

    # 3D plot
    if iter is None:
        title = f"{ftitle} 3D-probability\nK = {K}"
    else:
        title = f"{ftitle} 3D-probability\nK = {K}, iter = {iter}"

    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        projection="3d",
        title=title,
        xlabel="X",
        ylabel="Y",
        zlabel="Probability density",
    )
    ax.scatter3D(
        x,
        y,
        np.zeros(data.shape[0]),
        s=10,
        linewidths=1.0,
        marker="o",
        facecolor="None",
        edgecolors="darkblue",
        alpha=0.3,
        label="data",
    )
    ax.scatter3D(
        mu[:, 0],
        mu[:, 1],
        np.zeros(mu.shape[0]),
        s=100,
        linewidths=5,
        marker="x",
        c="r",
        label="centroids",
    )
    # ax.plot_wireframe(X, Y, probability)
    ax.contour3D(X, Y, probability, levels=50, cmap="rainbow", alpha=0.3)
    # plt.tight_layout()
    plt.grid(ls=":")
    plt.legend()

    return fig, ax


def em_algorithm(
    data,
    pi,
    mu,
    sigma,
    convergence=1e-3,
    max_iter=300,
    process_gif=False,
    ftitle=None,
):
    """
    EM algorithm.

    Parameters:
        data : ndarray (N, D)
            Input data from csv file.
        pi : ndarray (K,)
            Mixing coefficient.
        mu : ndarray (K, D)
            Average value.
        sigma : ndarray (K, D, D)
            Covariance matrix.
        convergence : float, Default=1e-3
            Condition of convergence.
        max_iter : int, Default=300
            Maximum number of iterations of the EM algorithm for a single run.

    Returns:
        likelihoods : list
            List of likelihood.
        pi : ndarray (K,)
            Mixing coefficient.
        mu : ndarray (K, D)
            Average value.
        sigma : ndarray (K, D, D)
            Covariance matrix.
    """
    N, D = data.shape
    K = mu.shape[0]  # n_clusters

    m_gauss, w_gauss = gmm(data, pi, mu, sigma)
    pre_likelihood = -np.inf
    likelihood = get_likelihood(m_gauss)
    likelihoods = [likelihood]
    iter = 0
    process1 = []
    process2 = []

    if process_gif:
        if D == 1:
            fig1, _ = plot_1d(data, pi, mu, sigma, ftitle, iter=iter)
        elif D == 2:
            fig1, _ = plot_2d(data, pi, mu, sigma, ftitle, iter=iter)
            fig2, _ = plot_3d(data, pi, mu, sigma, ftitle, iter=iter)

            fig2.canvas.draw()
            plt.close()
            im = np.array(fig2.canvas.renderer.buffer_rgba())
            img = Image.fromarray(im)
            process2.append(img)

        fig1.canvas.draw()
        plt.close()
        im = np.array(fig1.canvas.renderer.buffer_rgba())
        img = Image.fromarray(im)
        process1.append(img)

    while likelihood - pre_likelihood > convergence and iter < max_iter:
        # E step
        # burden rate
        gamma = w_gauss / m_gauss  # (k, N)

        # M step
        N_k = gamma.sum(axis=1)  # (k,)
        pi = N_k / N
        mu = np.sum(
            gamma.reshape(gamma.shape[0], gamma.shape[1], 1)
            * data
            / N_k.reshape(-1, 1, 1),
            axis=1,
        )
        diff_data = np.array([data - mu[i] for i in range(K)])  # (k, N, D)
        sigma = (
            gamma.reshape(gamma.shape[0], 1, gamma.shape[1])
            * diff_data.transpose(0, 2, 1)
            @ diff_data
        ) / N_k.reshape(-1, 1, 1)
        m_gauss, w_gauss = gmm(data, pi, mu, sigma)

        # calculate likelihood
        pre_likelihood = likelihood
        likelihood = get_likelihood(m_gauss)
        likelihoods.append(likelihood)
        iter += 1

        if process_gif:
            if D == 1:
                fig1, _ = plot_1d(data, pi, mu, sigma, ftitle, iter=iter)
            elif D == 2:
                fig1, _ = plot_2d(data, pi, mu, sigma, ftitle, iter=iter)
                fig2, _ = plot_3d(data, pi, mu, sigma, ftitle, iter=iter)

                fig2.canvas.draw()
                plt.close()
                im = np.array(fig2.canvas.renderer.buffer_rgba())
                img = Image.fromarray(im)
                process2.append(img)

            fig1.canvas.draw()
            plt.close()
            im = np.array(fig1.canvas.renderer.buffer_rgba())
            img = Image.fromarray(im)
            process1.append(img)

    return likelihoods, pi, mu, sigma, process1, process2


def main(args):
    """
    fname = "data1.csv"
    n_clusters = 2
    """
    fname = args.fname
    n_clusters = args.n_clusters

    # get current working directory
    path = os.path.dirname(os.path.abspath(__file__))
    ftitle, _ = os.path.splitext(fname)

    # load csv file and convert to ndarray
    data = pd.read_csv(os.path.join(path, "data", fname), header=None).values
    pi, mu, sigma = init(data, n_clusters)

    likelihoods, pi, mu, sigma, process1, process2 = em_algorithm(
        data, pi, mu, sigma, process_gif=True, ftitle=ftitle
    )

    # plot likelihood
    fig = plt.figure()
    fig.add_subplot(
        111,
        title=f"{ftitle} likelihood\nK = {n_clusters}",
        xlabel="Iteration",
        ylabel="Log likelihood",
    )
    plt.plot(range(0, len(likelihoods)), likelihoods)
    # plt.tight_layout()
    plt.grid(ls=":")
    plt.savefig(
        os.path.join(path, "result", f"{ftitle}_likelihood.png"), transparent=True
    )
    plt.show()

    # save process gif(2D)
    process1[0].save(
        os.path.join(path, "result", f"{ftitle}_process.gif"),
        save_all=True,
        append_images=process1[1:],
        loop=0,
        duration=100,
    )
    plt.close()

    # plot probability
    if data.shape[1] == 1:
        fig, ax = plot_1d(data, pi, mu, sigma, ftitle)
        plt.savefig(
            os.path.join(path, "result", f"{ftitle}_probability.png"), transparent=True
        )
        plt.show()

    elif data.shape[1] == 2:
        # save process gif(3D)
        process2[0].save(
            os.path.join(path, "result", f"{ftitle}_process3D.gif"),
            save_all=True,
            append_images=process2[1:],
            loop=0,
            duration=100,
        )
        plt.close()

        fig, ax = plot_2d(data, pi, mu, sigma, ftitle)
        plt.savefig(
            os.path.join(path, "result", f"{ftitle}_2Dprobability.png"),
            transparent=True,
        )
        plt.show()

        fig, ax = plot_3d(data, pi, mu, sigma, ftitle)
        plt.savefig(
            os.path.join(path, "result", f"{ftitle}_3Dprobability.png"),
            transparent=True,
        )
        plt.show()

        def update(i):
            """
            Move view point.

            Parameters:
                i : int
                    Number of frames.

            Returns:
                fig : matplotlib.figure.Figure
                    Figure viewed from angle designated by view_init function.
            """

            ax.view_init(elev=30.0, azim=3.6 * i)
            return fig

        # animate graph
        ani = animation.FuncAnimation(fig, update, frames=100, interval=100)
        ani.save(
            os.path.join(path, "result", f"{ftitle}_3Dprobability.gif"), writer="pillow"
        )
        plt.show()


if __name__ == "__main__":
    # process args
    parser = argparse.ArgumentParser(description="Fitting with GMM model.")
    parser.add_argument("fname", type=str, help="Load filename")
    parser.add_argument(
        "-n",
        "--n_clusters",
        type=int,
        help="The number of clusters (optional, Default=2)",
        default=2,
    )
    args = parser.parse_args()
    main(args)
