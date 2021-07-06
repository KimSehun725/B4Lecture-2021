import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys


def load_pickledata(data):
    """
    answer_models : answer label of model (the model that generated the output series)
    output : output series
    PI : initial probability
    A : transition probability matrix
    B : output probability
    """
    answer_models = np.array(data["answer_models"])  # 出力系列を生成したモデル（正解ラベル）
    output = np.array(data["output"])  # 出力系列
    models = data["models"]  # 定義済みHMM
    PI = np.array(models["PI"])  # 初期確率
    A = np.array(models["A"])  # 状態遷移確率行列
    B = np.array(models["B"])  # 出力確率
    return answer_models, output, PI, A, B


def forward_alg(answer_models, output, PI, A, B):
    """
    times : number of transitions
    n_series : number of series
    c : number of state
    m : number of output symbol
    q_i : i-th state
    o_k : k-th output symbol

    input
    ----------------------------------------------------------------------------------
    answer_models : answer label of model (the model that generated the output series)
    output : output series (number of observations,len_series : length of series)
    PI : initial probability (m,c,1)
    A : transition probability matrix (m,c,c) (a_i_j : probablity q_i => q_j)
    B : output probability (m,c,m) (b_i_k : probablity output o_k in q_i)

    output
    ---------------------------------
    array of HMM predicted by Forward algorithm
    """

    n_series, times = output.shape
    model_predict = np.empty(n_series)
    for i in range(n_series):
        alpha = PI[:, :, 0] * B[:, :, output[i, 0]]
        for t in range(1, times):
            alpha = np.sum(A.T * alpha.T, axis=1).T * B[:, :, output[i, t]]
        model_predict[i] = np.argmax(np.sum(alpha, axis=1))
    return model_predict


def viterbi_alg(answer_models, output, PI, A, B):
    n_series, times = output.shape
    model_predict = np.empty(n_series)
    for i in range(n_series):
        alpha = PI[:, :, 0] * B[:, :, output[i, 0]]
        for t in range(1, times):
            alpha = np.sum(A.T * alpha.T, axis=1).T * B[:, :, output[i, t]]
        model_predict[i] = np.argmax(np.max(alpha, axis=1))

    return model_predict


def calc_cm(answer, predict):
    model_number = list(set(answer)).sort()  # number of HMM
    cm = confusion_matrix(answer, predict, labels=model_number)
    cm = pd.DataFrame(cm, columns=model_number, index=model_number)
    return cm


def calc_accuracy(answer, predict):
    acc = np.sum(answer == predict) / len(answer) * 100
    return acc


def display_cm(answer, predict, algorithm, file):
    cm = calc_cm(answer, predict)
    acc = calc_accuracy(answer, predict)
    sns.heatmap(cm, annot=True, cbar=False, square=True, cmap="binary")
    plt.title(algorithm + " " + file + "\n(Acc. {}%)".format(acc))
    plt.xlabel("Predicted model")
    plt.ylabel("Actual model")
    plt.savefig(algorithm + "-" + file)
    plt.clf()
    plt.close


def main():
    args = sys.argv
    file = args[1]
    method = args[2]
    data = pickle.load(open(str(file) + ".pickle", "rb"))
    answer_models, output, PI, A, B = load_pickledata(data)
    if method == "f":
        algorithm = "foward"
        f_predict = forward_alg(answer_models, output, PI, A, B)
        display_cm(answer_models, f_predict, algorithm, file)
    elif method == "v":
        algorithm = "viterbi"
        v_predict = viterbi_alg(answer_models, output, PI, A, B)
        display_cm(answer_models, v_predict, algorithm, file)
    else:
        print("error,Enter f or v")


if __name__ == "__main__":
    main()
