import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import argparse
import csv
import random
import math


def csv_open(fname):
    """
    Function for opening csv file
    --------------------
    Parameters  :
    fname       : Name of the file
    --------------------
    Return      :
    file        : Data array in csv file
    """
    file = pd.read_csv(fname)
    file = np.array(file)
    return file


def pick_random(data, k):
    """
    Function for picking random data
    --------------------
    Parameters  :
    data        : data array
    k           : number of samples to pick
    --------------------
    Return      :
    value       : randomly picked data from data
    """
    index = random.sample(range(len(data)), k)
    value = data[index]
    return value


def k_means_2d(data, k):
    """
    Function for calculating initial state 
    calculated initial cluster info is fed to k_means_2d_reg function. 
    --------------------
    Parameters  :
    data        : data of csv file to calculate
    k           : coefficient for k_means algorithm
    """
    centroids = pick_random(data, k)
    x, y = data.T
    plt.plot(centroids.T[0], centroids.T[1], 'x',
             color='r', label='Initial centroid')
    cluster_n = []
    cluster = [k]

    for i in range(len(data)):
        dist = np.array([])
        for j in range(k):
            dist = np.append(dist, np.linalg.norm(data[i] - centroids[j]))
        # variable for which cluster every dot is in
        cluster_n = np.append(cluster_n, np.argmin(dist))

    plt.scatter(x, y, c=cluster_n)
    k_means_2d_reg(data, k, cluster_n, centroids)


def k_means_2d_reg(data, k, cluster_n, centroids):
    """
    Function for calculating k_means algorythm regressively 
    --------------------
    Parameters  :
    data        : Original data of csv file to calculate
    k           : coefficient for k_means algorithm
    cluster_n   : cluster information of each dot from previous calculation
    centroids   : centroid information of previous calculation
    """
    sum = np.zeros(shape=(k, 2))
    e = 0
    x, y = data.T
    for i in range(len(data)):
        for j in range(k):
            if cluster_n[i] == j:
                sum[j] += data[i]

    n_points = np.array([])
    center = np.zeros(shape=(k, 2))
    for i in range(k):
        n_points = np.append(n_points, np.count_nonzero(cluster_n == i))
        center[i][0] = sum[i][0]/n_points[i]
        center[i][1] = sum[i][1]/n_points[i]

    center_x = center.T[0]
    center_y = center.T[1]

    cluster_n = []

    for i in range(len(data)):
        dist = np.array([])
        for j in range(k):
            dist = np.append(dist, np.linalg.norm(data[i] - center[j]))
        cluster_n = np.append(cluster_n, np.argmin(dist))

    for j in range(k):
        e += np.linalg.norm(center[j] - centroids[j], ord=2)
    e_avg = e/k

    if e_avg == 0.0:
        plt.scatter(x, y, c=cluster_n)
        plt.scatter(center_x, center_y, c='r', label='Calculated centroids')
        plt.legend()
        plt.savefig("plot.png")
        return 0
    else:

        k_means_2d_reg(data, k, cluster_n, center)


def k_means_3d(data, k):
    """
    Function for calculating initial state 
    calculated initial cluster info is fed to k_means_3d_reg function. 
    --------------------
    Parameters  :
    data        : data of csv file to calculate
    k           : coefficient for k_means algorithm
    """
    centroids = pick_random(data, k)
    x, y, z = data.T
    cluster_n = []
    cluster = [k]

    for i in range(len(data)):
        dist = np.array([])
        for j in range(k):
            dist = np.append(dist, np.linalg.norm(data[i] - centroids[j]))
        # variable for which cluster every dot is in
        cluster_n = np.append(cluster_n, np.argmin(dist))

    k_means_3d_reg(data, k, cluster_n, centroids)


def k_means_3d_reg(data, k, cluster_n, centroids):
    """
    Function for calculating k_means algorythm regressively 
    --------------------
    Parameters  :
    data        : Original data of csv file to calculate
    k           : coefficient for k_means algorithm
    cluster_n   : cluster information of each dot from previous calculation
    centroids   : centroid information of previous calculation
    """
    sum = np.zeros(shape=(k, 3))
    e = 0
    x, y, z = data.T
    for i in range(len(data)):
        for j in range(k):
            if cluster_n[i] == j:
                sum[j] += data[i]

    n_points = np.array([])
    center = np.zeros(shape=(k, 3))
    for i in range(k):
        n_points = np.append(n_points, np.count_nonzero(cluster_n == i))
        center[i][0] = sum[i][0]/n_points[i]
        center[i][1] = sum[i][1]/n_points[i]
        center[i][2] = sum[i][2]/n_points[i]

    center_x = center.T[0]
    center_y = center.T[1]
    center_z = center.T[2]

    cluster_n = []

    for i in range(len(data)):
        dist = np.array([])
        for j in range(k):
            dist = np.append(dist, np.linalg.norm(data[i] - center[j]))
        cluster_n = np.append(cluster_n, np.argmin(dist))

    for j in range(k):
        e += np.linalg.norm(center[j] - centroids[j], ord=2)
    e_avg = e/k

    if e_avg == 0.0:
        fig = plt.figure()
        ax = Axes3D(fig)

        def rotate(angle):
            ax.view_init(azim=angle)
        ax.scatter3D(x, y, z, c=cluster_n)
        ax.scatter(center_x, center_y, center_z,
                   c='r', label='Calculated centroids')
        plt.legend()
        rot_animation = animation.FuncAnimation(
            fig, rotate, frames=100, interval=50)
        rot_animation.save('rotation3D.gif', dpi=80)
        return 0
    else:
        k_means_3d_reg(data, k, cluster_n, center)


def main():
    parser = argparse.ArgumentParser(
        description='Program for plotting regression equation.\nFile name and order for regression equation are required.')
    parser.add_argument("-f", dest="filename", help='Filename', required=True)
    parser.add_argument("-k", dest="k", type=int,
                        help='number for clustering', required=False, default=4)
    args = parser.parse_args()
    data = csv_open(args.filename)
    k = args.k
    dimension = np.shape(data)[1]

    if dimension == 2:
        x, y = data.T
        k_means_2d(data, k)
    elif dimension == 3:

        k_means_3d(data, k)


if __name__ == "__main__":
    main()
