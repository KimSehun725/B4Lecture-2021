import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
from PIL import Image

import argparse
import sys

from sklearn.decomposition import PCA


# ファイル読み込み
def file(f_name):
    """
    paramerter
    ---
    num:int
        file number

    return
    ---
    df:pandas.core.frame.DataFrame
       csv.data
    data:numpy.ndarray
        csv data
    """
    # if f_name in[1,2,3]:
    f_name = "/Users/nobat/b4rinkou/B4Lecture-2021/ex_6/data" + str(f_name) + ".csv"
    df = pd.read_csv(f_name)
    data = df.values
    return df, data


# 2次元散布図
def scatter_2d(data, vec, save):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    vec:numpy.ndarray

    save:str
         save file name

    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    x = data[:, 0]
    y = data[:, 1]
    ax.plot(x, y, label="data", marker="o", linestyle="None")

    for i in range(vec.shape[0]):
        x = np.linspace(-1, 1, 2)
        y = vec[i, 1] / vec[i, 0] * x
        ax.plot(x, y, label=str(i + 1))

    # ラベル作成
    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    plt.title("data", fontsize=18)
    plt.grid()
    plt.legend()  # 凡例の追加
    plt.tight_layout()
    plt.savefig(save)
    plt.show()


# 3次元散布図＆回転
def render_frame(data, vec, angle):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    ax.set_zlabel("z", fontsize=18)
    ax.plot(x, y, z, label="data", marker="o", linestyle="None")

    for i in range(vec.shape[0]):
        x = np.linspace(-1, 1, 2)
        y = vec[i, 1] / vec[i, 0] * x
        z = vec[i, 2] / vec[i, 0] * x
        ax.plot(x, y, z, label=str(i + 1))

    ax.view_init(30, angle * 2)
    plt.title("data2", fontsize=18)
    plt.grid()
    plt.legend()  # 凡例の追加
    plt.tight_layout()
    plt.close()
    buf = BytesIO()
    fig.savefig(buf, bbox_inches="tight", pad_inches=0.0)
    return Image.open(buf)


# 3次元画像gif保存
def scatter_3d(data, vec, savename):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    save:str
         save file name
    """
    images = [render_frame(data, vec, angle) for angle in range(45)]
    images[0].save(
        savename, save_all=True, append_images=images[1:], duration=100, loop=0
    )


# 主成分分析
def mpca(data, std=1):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data

    returns
    ---
    con_ratio:numpy.ndarray
              contribution rate
    sort_eig_val:numpy.ndarray
                 sorted eigenvalue
    """
    # 標準化
    if std == 1:
        data = (data - data.mean()) / np.var(data, axis=0)

    # 共分散行列
    cov_vec = np.cov(data, rowvar=0)

    # 固有値と固有ベクトル
    eig_val, eig_vec = np.linalg.eig(cov_vec)

    # 固有値を大きい順に並べる
    eig_idx = np.argsort(eig_val)[::-1]
    sort_eig_val = eig_val[eig_idx]
    sort_eig_vec = eig_vec[:, eig_idx]  # 固有ベクトル=主成分

    # 寄与率を求める
    con_ratio = sort_eig_val / np.sum(sort_eig_val)

    # 主成分得点
    pcc = np.dot(data, eig_vec)

    return con_ratio, sort_eig_vec, pcc


# 累積寄与率
def cumulative(con_ratio):
    """
    paramerters
    --
    con_ratio:numpy.ndarray
              contribution rate

    returns
    ---
    cum:numpy.ndarray
            cumlative contribution ratio(%)
    """
    m = con_ratio.shape[0]
    cum = np.zeros(m)
    c = 0
    for i in range(m):
        c += con_ratio[i] * 100
        cum[i] = c
    return cum


# 次元圧縮
def dim_compressiom(pcc):
    plt.scatter(-pcc[:, 0], -pcc[:, 1])
    plt.title("principal component")
    plt.xlabel("Z1")
    plt.ylabel("Z2")
    plt.title("data", fontsize=18)
    plt.grid()
    plt.tight_layout()
    plt.savefig("dimdata.png")
    plt.show()


# 比較用
def pcan(data):
    pca = PCA()
    pca.fit(data)
    transformed = pca.fit_transform(data)

    # 主成分をプロットする
    plt.scatter(transformed[:, 0], transformed[:, 1])
    plt.title("principal component")
    plt.xlabel("Z1")
    plt.ylabel("Z2")
    plt.title("data", fontsize=18)
    plt.grid()
    plt.tight_layout()
    # plt.savefig("PCA_sample.png")
    plt.show()


def main(args):
    fnum = args.fnum
    per = args.per
    savename = args.savename
    df, data = file(fnum)
    num, dim = data.shape
    con, vec, pcc = mpca(data)
    cum = cumulative(con)

    m = np.min(np.where(cum >= per))
    print(str(m + 1) + "次元" + str(cum[m]) + "%")

    dim_compressiom(pcc)

    if dim == 2:
        scatter_2d(data, vec, savename)
    elif dim == 3:
        scatter_3d(data, vec, savename)

    else:
        print("error:over dimension")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fnum", default="2")
    parser.add_argument("--savename", default="data2.png")
    parser.add_argument("--per", default=90)
    args = parser.parse_args()

    main(args)
