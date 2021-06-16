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


def plot2d(data):
    """
    Function for plotting 2D data
    --------------------
    Parameters  :
    data        : Data array to plot
    """
    x, y = data.T
    plt.scatter(x, y)
    plt.savefig("original2d.png")


def plot3d(data):
    """
    Function for plotting 3D data
    --------------------
    Parameters  :
    data        : Data array to plot
    """
    x, y, z = data.T
    fig = plt.figure()
    ax = Axes3D(fig)

    def rotate(angle):
        ax.view_init(azim=angle)
    ax.scatter3D(x, y, z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # plt.legend()
    rot_animation = animation.FuncAnimation(
        fig, rotate, frames=100, interval=30)
    rot_animation.save('rotation3D.gif', dpi=80)


def plot_contrib(contribution_rate):
    plt.plot(contribution_rate)
    plt.title('contribution rate')
    plt.xlabel('eigen vectors')
    plt.ylabel('contribution rate')
    plt.savefig("contribution_rate.png")
    plt.clf()


def pcd(data):
    """
    Function for calculating principal component
    --------------------
    Parameters          :
    data                : Array of data to analyse
    --------------------
    Return              :
    eig_val             : Eigen-value of data 
    eig_vec             : Eogen-vector of data
    contribution_rate   : Contribution rate of each eigen-vector
    """
    avg = np.average(data, axis=0)
    X = data - avg
    std_dev = np.std(data, axis=0)
    X /= std_dev
    cov = np.dot(X.T, X) / len(X)
    eig_val, eig_vec = np.linalg.eig(cov)
    eig_vec = eig_vec[:, np.argsort(eig_val)[::-1]]
    eig_val = np.sort(eig_val)[::-1]
    contribution_rate = np.array([eig_val[i]/sum(eig_val)
                                  for i in range(len(eig_val))])

    return eig_val, eig_vec, contribution_rate


def main():
    parser = argparse.ArgumentParser(
        description='Program for plotting regression equation.\nFile name and order for regression equation are required.')
    parser.add_argument("-f", dest="filename", help='Filename', required=True)
    args = parser.parse_args()
    data = csv_open(args.filename)
    dimension = np.shape(data)[1]
    eig_val, eig_vec, contribution_rate = pcd(data)

    if dimension == 2:
        x = np.arange(min(data[:, 0]), max(data[:, 0]), 0.01)
        y1 = eig_vec[1][0] / eig_vec[0][0] * x
        y2 = eig_vec[1][1] / eig_vec[0][1] * x
        plt.plot(x, y1, label="contribution rate:{:.3}".format(
            contribution_rate[0]))
        plt.plot(x, y2, label="contribution rate:{:.3}".format(
            contribution_rate[1]))
        plt.legend()
        plot2d(data)

    elif dimension == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        original_x, original_y, original_z = data.T
        x = np.arange(min(data[:, 0]), max(data[:, 0]), 0.01)
        y1 = eig_vec[1][0] / eig_vec[0][0] * x
        z1 = eig_vec[2][0] / eig_vec[0][0] * x
        y2 = eig_vec[1][1] / eig_vec[0][1] * x
        z2 = eig_vec[2][1] / eig_vec[0][1] * x
        y3 = eig_vec[1][2] / eig_vec[0][2] * x
        z3 = eig_vec[2][2] / eig_vec[0][2] * x
        ax.plot(x, y1, z1, label="contribution rate:{:.3}".format(
            contribution_rate[0]), c='r')
        ax.plot(x, y2, z2, label="contribution rate:{:.3}".format(
            contribution_rate[1]))
        ax.plot(x, y3, z3, label="contribution rate:{:.3}".format(
            contribution_rate[2]))

        def rotate(angle):
            ax.view_init(azim=angle)
        ax.scatter3D(original_x, original_y, original_z)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        """
        ax.set_xlim(-2.5,2.5)
        ax.set_ylim(-2.5,2.5)
        ax.set_zlim(-2.5,2.5)
        """
        plt.legend()
        rot_animation = animation.FuncAnimation(
            fig, rotate, frames=200, interval=50)
        rot_animation.save('rotation3D.gif', dpi=80)
        plt.clf()
        plt.scatter(np.dot(data, eig_vec[0]), np.dot(data, eig_vec[1]))
        plt.savefig("transformed.png")

        # plot3d(data)

    elif dimension > 3:
        plot_contrib(contribution_rate)
        plt.scatter(np.dot(data, eig_vec[0]), np.dot(data, eig_vec[1]))
        plt.xlabel('1st eigen vector')
        plt.ylabel('2nd eigen vector')
        plt.savefig("transformed.png")
        sum_contrib = 0
        for i in range(len(contribution_rate)):
            sum_contrib += contribution_rate[i]
            if sum_contrib > 0.9:
                print("dimention needed to express contribution rate > 90% :  ", i+1)
                break
        for i in range(len(eig_val)-1):
            print("inner product of eigen vector[%d],[%d]" % (i, i +1), round(np.dot(eig_vec[i], eig_vec[i+1].T), -10))



if __name__ == "__main__":
    main()
