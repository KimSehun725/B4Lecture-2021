import os
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler


def pca(data):
    """
    input
    data : ndarray
           target data for PCA

    output
    eig_vec  : ndarray
               the eigenvectors of the input data covariance matrix
    con_rate : ndarray
               contribution rate
    """

    # data standardization
    data_std = StandardScaler().fit_transform(data)
    # calc covariance matrix
    cov_matrix = np.cov(data_std, rowvar=False)
    # calc eigenvalues and eigenvectors
    eig_val, eig_vec = np.linalg.eigh(cov_matrix)
    
    # sort data
    eig_val = np.sort(np.abs(eig_val))[:: -1]
    eig_vec = eig_vec[:, np.argsort(np.abs(eig_val))[::-1]]
    # calc contribution rate
    con_rate = eig_val / sum(eig_val)
    
    return eig_vec, con_rate


def main():
    # argments settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input data file path")
    args = parser.parse_args()

    # check output dir exist
    if not os.path.exists('./out'):
        os.makedirs('./out')

    # loading data and calc eigenvector
    data = pd.read_csv(args.input, header=None).to_numpy()
    eig_vec, con_rate = pca(data)   
    
    # graphic setting
    x = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
    label_top = "Contribution rate: "
    colormap = mpl.cm.tab10.colors
    
    # in the case of 2 dimensions
    if data.shape[1] == 2:
        fig = plt.figure()
        
        plt.scatter(data[:, 0], data[:, 1], c="w", edgecolor=colormap[0], label=f"Data{data.shape[1] - 1}")
        plt.plot(x, eig_vec[0, 0]/eig_vec[0, 1]*x, label=f"{label_top}{con_rate[0]:.3f}", color=colormap[1])
        
        plt.plot(x, eig_vec[1, 0]/eig_vec[1, 1]*x, label=f"{label_top}{con_rate[1]:.3f}", color=colormap[2])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        
        plt.show(block=True)
        fig.savefig('./out/data1.png')

    # in the case of 3 dimensions
    elif data.shape[1] == 3:
        fig = plt.figure()
        ax = Axes3D(fig)

        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c="w", edgecolor=colormap[0], label=f"Data{data.shape[1] - 1}")
        ax.plot(x, eig_vec[0, 1]/eig_vec[0, 0]*x, eig_vec[0, 2]/eig_vec[0, 0]*x, label=f"{label_top}{con_rate[0]:.3f}", color=colormap[1])
        ax.plot(x, eig_vec[1, 1]/eig_vec[1, 0]*x, eig_vec[1, 2]/eig_vec[1, 0]*x, label=f"{label_top}{con_rate[1]:.3f}", color=colormap[2])
        ax.plot(x, eig_vec[2, 1]/eig_vec[2, 0]*x, eig_vec[2, 2]/eig_vec[2, 0]*x, label=f"{label_top}{con_rate[2]:.3f}", color=colormap[3])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        
        plt.show(block=True)
        fig.savefig('./out/data2_3d.png')

        # calc and plot compressed data
        comp_data = np.dot(data, eig_vec.T)
        fig = plt.figure()
        
        plt.scatter(comp_data[:, 0], comp_data[:, 1])
        plt.title('Dimensional Compression')
        
        plt.show(block=True)
        fig.savefig('./out/data2_2dcomp.png')

    # in the case of 4 dimensions or more
    else:
        flag = 1
        tot_rate = 0
        rate_list = []
        for i in range(len(con_rate)):
            tot_rate += con_rate[i]
            rate_list.append(tot_rate)
            # check contribution rate
            if tot_rate >= 0.9 and flag == 1:
                dim = i + 1
                flag = 0
        
        fig = plt.figure()
        plt.plot(range(1, len(con_rate)+1), rate_list)
        plt.hlines([0.9], 1, 100, color="orange", linestyles='dashed')
        plt.vlines([dim], 0, 1, color="orange", linestyles='dashed')
        plt.show(block=True)
        fig.savefig('./out/data3.png')

        print(f'Can be compressed up to {dim} dimensions! (The original data is {data.shape[1]} dimensions.)')


if __name__ == '__main__':
    main()
