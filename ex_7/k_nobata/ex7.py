import argparse

import numpy as np
import pandas as pd
import random

from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# ファイル読み込み
def file(f_name):
    """
    paramerter
    ---
    num:str
        file name

    return
    ---
    df:pandas.core.frame.DataFrame
       csv.data
    data:numpy.ndarray
        csv data
    """
    df = pd.read_csv(f_name)
    data = df.values
    return df, data


# 対数尤度関数のプロット
def logplot(log_list, save):
    """
    log_list:list
            log likelihood function
    save:str
         save name
    """
    num = str(save)[0]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(log_list)
    ax.set_xlabel("count", fontsize=18)
    ax.set_ylabel("log likelihood function", fontsize=18)
    plt.title("data" + num, fontsize=18)
    plt.grid()
    plt.tight_layout()
    if type(save) == str:
        logsave = "log" + save
        plt.savefig(logsave)
    plt.show()


# 1次元散布図
def scatter_1d(data, clu, vec, cov, pi, save=0):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    clu:numpy.ndarray
        cluster
    vec:numpy.ndarray
        mean vector
    cov:numpy.ndarray
        covariance matrix
    pi:numpy.ndarray
       mixing coefficient
    save:str
        save file name
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    num = str(save)[0]
    K = len(vec)

    # クラスタプロット
    cmap = plt.get_cmap("tab10")
    for i in range(K):
        cdata = data[clu == i]
        x = cdata
        y = np.zeros(x.shape[0])
        # 計算値とデータをプロッ
        ax.plot(x, y, label=i, marker="o", linestyle="None", color=cmap(i))
    x = vec
    y = np.zeros(K)
    ax.plot(x, y, label="centroids", marker="x", linestyle="None", color=cmap(i + 1))

    # ガウス分布
    pos = np.linspace(np.min(data) - 1, np.max(data) + 1, 100)
    y = np.zeros(100)
    for k in range(K):
        y += pi[k] * multivariate_normal.pdf(pos, vec[k], cov[k])
    ax.plot(pos, y, label="GMM")

    # ラベル作成
    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("gaussian distribution", fontsize=18)
    plt.title("data" + num, fontsize=18)
    plt.grid()
    plt.legend()  # 凡例の追加
    plt.tight_layout()
    if type(save) == str:
        gmmsave = "gmm" + save
        plt.savefig(gmmsave)
    plt.show()


# 2次元散布図
def scatter_2d(data, clu, vec, cov, pi, save=0):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    clu:numpy.ndarray
        cluster
    vec:numpy.ndarray
        mean vector
    cov:numpy.ndarray
        covariance matrix
    pi:numpy.ndarray
       mixing coefficient
    save:str
        save file name
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    num = str(save)[0]
    K = len(vec)

    # クラスタプロット
    cmap = plt.get_cmap("tab10")
    for i in range(K):
        cdata = data[clu == i]
        x = cdata[:, 0]
        y = cdata[:, 1]
        ax.plot(x, y, label=i, marker="o", linestyle="None", color=cmap(i))
    x = vec[:, 0]
    y = vec[:, 1]
    ax.plot(x, y, label="centroids", marker="x", linestyle="None", color=cmap(i + 1))

    # 等高線
    posx = np.linspace(np.min(data[:, 0]) - 1, np.max(data[:, 0]) + 1, 100)
    posy = np.linspace(np.min(data[:, 1]) - 1, np.max(data[:, 1]) + 1, 100)
    posx, posy = np.meshgrid(posx, posy)
    pos = np.dstack([posx, posy])
    z = np.array([np.squeeze(gmm(i_pos, vec, cov, pi)[1], 0) for i_pos in pos])

    plt.contour(posx, posy, z)
    plt.colorbar()

    # ラベル作成
    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    plt.title("data" + num, fontsize=18)
    plt.grid()
    plt.legend()  # 凡例の追加
    plt.tight_layout()
    if type(save) == str:
        gmmsave = "gmm" + save
        plt.savefig(gmmsave)
    plt.show()


# 初期値決定
# ミニマックス法
def minimax(data, K):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    K:int
      the number of cluster

    return
    ----
    cen:numpy.ndarray
         center of cluster
    """
    num, dim = data.shape
    cidx = []  # 中心のインデックス
    cidx = np.append(cidx, random.randint(0, num - 1))
    dis = np.zeros((K, num))
    cen = np.zeros((K, dim))
    for k in range(K):
        cen[k] = data[int(cidx[k])]
        r = np.sum((data - data[int(cidx[k])]) ** 2, axis=1)  # 距離計算
        dis[k] = r  # 距離保存

        cidx = np.append(cidx, np.argmax(np.min(dis[: k + 1], axis=0)))  # 距離最大の次の中心
    # clu = np.argmin(dis, axis=0)
    return cen


# kmeanアルゴリズム
def kmean(data, K, cen):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    K:int
      the number of cluster
    cen:numpy.ndarray
         center of cluster
    clu:numpy.ndarray
        cluster

    return
    ----
    newcen:numpy.ndarray
           new center of cluster
    clu:numpy.ndarray
        cluster
    """
    num, dim = data.shape
    dis = np.zeros((K, num))
    newcen = np.zeros((K, dim))
    while True:
        for k in range(0, K):
            r = np.sum((data - cen[k]) ** 2, axis=1)  # 距離計算
            dis[k] = r  # 距離保存

        clu = np.argmin(dis, axis=0)

        for i in range(0, K):
            newcen[i] = data[clu == i].mean(axis=0)

        if np.allclose(cen, newcen) is True:
            break
        cen = newcen
    return newcen, clu


# 初期値決定
def ini(data, K):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    K:int
      the number of cluster

    returns
    ---
    vec:numpy.ndarray
        mean vector
    cov:numpy.ndarray
        covariance matrix
    pi:numpy.ndarray
       mixing coefficient

    """
    num, dim = data.shape
    cen = minimax(data, K)
    vec, clu = kmean(data, K, cen)
    pi = np.zeros(K)
    cov = np.zeros((K, dim, dim))

    for k in range(K):
        pi[k] = data[clu == k].shape[0]
        cov[k] = np.cov(data[clu == k].T)

    pi = pi / np.sum(pi)
    return vec, cov, pi


# 初期値設定(適当)
def setInitial(data, K):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    K:int
      the number of cluster

    returns
    ---
    vec:numpy.ndarray
        mean vector
    cov:numpy.ndarray
        covariance matrix
    pi:numpy.ndarray
       mixing coefficient

    """
    D = data.shape[1]
    vec = np.random.randn(K, D)
    cov = np.array([np.eye(D) for i in range(K)])
    pi = np.array([1 / K for i in range(K)])
    return vec, cov, pi


# データすべての多次元ガウス分布
def gauss_all(Data, vec, cov):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    vec:numpy.ndarray
        mean vector
    cov:numpy.ndarray
        covariance matrix

    return
    ---
    Nk:numpy.ndarray

    """
    num, dim = Data.shape
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    Nk = np.zeros(num)

    for i in range(num):
        a = ((2 * np.pi) ** (0.5 * dim)) * (det ** 0.5)
        b = -0.5 * (Data[i] - vec)[None, :] @ inv @ (Data[i] - vec)
        Nk[i] = np.exp(b) / a

    return Nk


# 混合ガウス分布
def gmm(Data, Vec, Cov, Pi):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    vec:numpy.ndarray
        mean vector
    cov:numpy.ndarray
        covariance matrix
    pi:numpy.ndarray
       mixing coefficient

    return
    --
    N:numpy.ndarray
    np.sum(N, axis=0)[None,:]:numpy.ndarray

    """
    k = len(Vec)
    N = np.array([Pi[i] * gauss_all(Data, Vec[i], Cov[i]) for i in range(k)])
    return N, np.sum(N, axis=0)[None, :]


# 対数尤度関数
def log_likelihood(Data, Vec, Cov, Pi):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    vec:numpy.ndarray
        mean vector
    cov:numpy.ndarray
        covariance matrix
    pi:numpy.ndarray
       mixing coefficient

    return
    ---
    np.sum(logs):numpy.ndarray
                 log likelihood function
    """
    num, dim = Data.shape
    _, N_sum = gmm(Data, Vec, Cov, Pi)
    logs = np.array([np.log(N_sum[0][i]) for i in range(num)])
    return np.sum(logs)


# EMアルゴリズム
def Em(data, vec, cov, pi, eps):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    vec:numpy.ndarray
        mean vector
    cov:numpy.ndarray
        covariance matrix
    pi:numpy.ndarray
       mixing coefficient
    eps:float
        threshold

    return
    ---
    gamma:numpy.ndarray
         burden rate
    vec:numpy.ndarray
        mean vector
    cov:numpy.ndarray
        covariance matrix
    pi:numpy.ndarray
       mixing coefficient
    """
    K = vec.shape[0]
    num, dim = data.shape
    count = 0
    log_list = []

    while True:
        old_log = log_likelihood(data, vec, cov, pi)
        N, N_sum = gmm(data, vec, cov, pi)
        cov = np.zeros((K, dim, dim))
        # Eステップ
        # 負担率更新
        gamma = N / N_sum

        # Mステップ

        # 平均ベクトル更新
        vec = (gamma @ data) / np.sum(gamma, axis=1)[:, None]

        # 分散更新
        for k in range(K):
            for n in range(num):
                dis = data[n] - vec[k]
                cov[k] += gamma[k][n] * dis[:, None] @ dis[None, :]

            cov[k] = cov[k] / np.sum(gamma[k])

        # 混合係数
        pi = np.sum(gamma, axis=1) / num

        new_log = log_likelihood(data, vec, cov, pi)
        log_dif = old_log - new_log
        log_list = np.append(log_list, log_dif)

        # 収束確認
        if np.abs(log_dif) < eps:
            return count, gamma, vec, cov, pi, log_list
        else:
            count += 1
            old_log = new_log


def main(args):
    fname = args.fname
    save = args.savename
    K = args.K

    df, data = file(fname)
    num, dim = data.shape
    eps = 0.01
    vec0, cov0, pi0 = ini(data, K)
    count, gamma, vec, cov, pi, log_list = Em(data, vec0, cov0, pi0, eps)

    clu = np.argmax(gamma, axis=0)
    if dim == 1:
        scatter_1d(data, clu, vec, cov, pi, save)
    elif dim == 2:
        scatter_2d(data, clu, vec, cov, pi, save)
    else:
        "error: dimension"
    logplot(log_list, save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fname", default="/Users/nobat/b4rinkou/B4Lecture-2021/ex_7/data3.csv"
    )
    parser.add_argument("--K", default=2)
    parser.add_argument("--savename", default="3data.png")
    args = parser.parse_args()

    main(args)
