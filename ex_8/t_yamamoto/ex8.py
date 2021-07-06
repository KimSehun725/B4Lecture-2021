# include flake8, black

import argparse
import os
import time

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score


def forward(output, PI, A, B):
    """
    Predict HMM by forward algorithm.

    Parameters:
        output : ndarray (n, n)
            Output series.
        PI : ndarray (k, s, 1)
            Initial probability.
        A : ndarray (k, s, s)
            State transition probability matrix.
        B : ndarray (k, s, k)
            Output probability.

    Returns:
        predict : ndarray (n,)
            HMM predicted by forward algorithm.

    Note:
        n : The number of output series.
        k : The number of HMM.
        s : The number of states.
    """
    n_out, l_out = output.shape
    predict = np.empty(n_out)
    for i in range(n_out):
        alpha = PI[:, :, 0] * B[:, :, output[i, 0]]
        for j in range(1, l_out):
            alpha = np.sum(A.T * alpha.T, axis=1).T * B[:, :, output[i, j]]
        predict[i] = np.argmax(np.sum(alpha, axis=1))
    return predict


def viterbi(output, PI, A, B):
    """
    Predict HMM by viterbi algorithm.

    Parameters:
        output : ndarray (n, n)
            Output series.
        PI : ndarray (k, s, 1)
            Initial probability.
        A : ndarray (k, s, s)
            State transition probability matrix.
        B : ndarray (k, s, k)
            Output probability.

    Returns:
        predict : ndarray (n,)
            HMM predicted by viterbi algorithm.

    Note:
        n : The number of output series.
        k : The number of HMM.
        s : The number of states.
    """
    n_out, l_out = output.shape
    predict = np.empty(n_out)
    for i in range(n_out):
        alpha = PI[:, :, 0] * B[:, :, output[i, 0]]
        for j in range(1, l_out):
            alpha = np.max(A.T * alpha.T, axis=1).T * B[:, :, output[i, j]]
        predict[i] = np.argmax(np.max(alpha, axis=1))
    return predict


def display_cm(answer, predict, title):
    """
    Display confusion matrix.

    Parameters:
        answer : ndarray (n,)
            Correct label.
        predict : ndarray (n,)
            HMM predicted by forward or viterbi algorithm.
        title : str
            Algorithm name for figure title.

    Note:
        n : The number of output series.
    """
    label = list(map(lambda x: x + 1, list(set(answer))))
    cm = confusion_matrix(answer, predict)
    cm = pd.DataFrame(cm, columns=label, index=label)
    acc = accuracy_score(answer, predict) * 100
    sns.heatmap(cm, annot=True, cbar=False, square=True, cmap="binary")
    plt.title(f"{title}\n(Acc. {acc:.1f}%)")
    plt.xlabel("Predicted model")
    plt.ylabel("Actual model")


def main(args):
    # fname = "data1.pickle"
    fname = args.fname

    # get current working directory
    path = os.path.dirname(os.path.abspath(__file__))

    ftitle, _ = os.path.splitext(fname)
    save_fname = os.path.join(path, "result", f"heatmap_{ftitle}.png")

    # load pickle data
    data = pickle.load(open(os.path.join(path, "data", fname), "rb"))
    # data
    # ├─answer_models # 出力系列を生成したモデル（正解ラベル）
    # ├─output # 出力系列
    # └─models # 定義済みHMM
    #   ├─PI # 初期確率
    #   ├─A # 状態遷移確率行列
    #   └─B # 出力確率

    answer_models = np.array(data["answer_models"])
    output = np.array(data["output"])
    PI = np.array(data["models"]["PI"])
    A = np.array(data["models"]["A"])
    B = np.array(data["models"]["B"])

    """
    print(answer_models.shape)
    print(output.shape)
    print(PI.shape)
    print(A.shape)
    print(B.shape)
    """

    # HMM
    start = time.time()
    forward_predict = forward(output, PI, A, B)
    print(f"elapsed time for forward algorithm: {time.time() - start:.4f}s")
    start = time.time()
    viterbi_predict = viterbi(output, PI, A, B)
    print(f"elapsed time for viterbi algorithm: {time.time() - start:.4f}s")

    # display result
    fig = plt.figure()
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(1, 2, 1)
    display_cm(answer_models, forward_predict, title="Forward algorithm")
    plt.subplot(1, 2, 2)
    display_cm(answer_models, viterbi_predict, title="Viterbi algorithm")
    fig.suptitle(ftitle, fontsize=18, y=0.9)
    # fig.tight_layout()
    plt.savefig(save_fname, transparent=True)
    plt.show()


if __name__ == "__main__":
    # process args
    parser = argparse.ArgumentParser(description="HMM Model prediction.")
    parser.add_argument("fname", type=str, help="Load filename")
    args = parser.parse_args()
    main(args)
