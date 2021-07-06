import argparse
import sys

import time

import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


# ファイル読み込み
def file(fnum):
    """
    paramerter
    ---
    fnum:int
        file number

    return
    ---
    output:numpy.ndarray(100,100)
           output series
    A:numpy.ndarray (5,3,3) or (5,5,5)
      transition probability matrix
    PI:numpy.ndarray(5,3) or (5,5)
      initial probability
    B:numpy.ndarray(5,3,5) or (5,5,5)
      output probability
    answer:numpy.ndarray (100)
           answer models
    """
    if fnum in [1, 2, 3, 4]:
        data = pickle.load(
            open(
                "/Users/nobat/b4rinkou/B4Lecture-2021/ex_8/data"
                + str(fnum)
                + ".pickle",
                "rb",
            )
        )
    else:
        print("error:file not found")
        sys.exit()

    # dataframe type to numpy type
    output = np.array(data["output"])
    A = np.array(data["models"]["A"])
    PI = np.array(data["models"]["PI"]).squeeze()
    B = np.array(data["models"]["B"])
    answer = np.array(data["answer_models"])

    return output, A, PI, B, answer


# Forwardアルゴリズム
def forward(output, A, PI, B):
    """
    paramerters
    ---
    output:numpy.ndarray(100, 100)
           output series
    A:numpy.ndarray (5, 3, 3) or (5, 5, 5)
      transition probability matrix
    PI:numpy.ndarray(5, 3) or (5, 5)
      initial probability
    B:numpy.ndarray(5, 3, 5) or (5, 5, 5)
      output probability

    return
    ---
    predict:numpy.ndarray (100)
            predicted models
    """
    NUM, TRA = output.shape
    predict = np.zeros(NUM)

    for i in range(NUM):  # each data
        alpha = PI * B[:, :, output[i, 0]]  # initial value
        for j in range(1, TRA):  # transitions
            alpha = B[:, :, output[i, j]] * np.sum(A.T * alpha.T, axis=1).T
        predict[i] = np.argmax(np.sum(alpha, axis=1))

    return predict


# Viterbiアルゴリズム
def viterbi(output, A, PI, B):
    """
    paramerters
    ---
    output:numpy.ndarray(100, 100)
           output series
    A:numpy.ndarray (5, 3, 3) or (5, 5, 5)
      transition probability matrix
    PI:numpy.ndarray(5, 3) or (5, 5)
      initial probability
    B:numpy.ndarray(5, 3, 5) or (5, 5, 5)
      output probability

    return
    ---
    predict:numpy.ndarray (100)
            predicted models
    """
    NUM, TRA = output.shape
    predict = np.zeros(NUM)

    for i in range(NUM):  # each data
        alpha = PI * B[:, :, output[i, 0]]  # initial value
        for j in range(1, TRA):  # transitions
            alpha = B[:, :, output[i, j]] * np.max(A.T * alpha.T, axis=1).T
        predict[i] = np.argmax(np.max(alpha, axis=1))

    return predict


# 混合行列
def calc_cm(answer, predict):
    """
    paramerters
    ---
    answer:numpy.ndarray (100)
           answer models
    predict:numpy.ndarray (100)
            predicted models

    return
    ---
    cm:pandas.core.frame.DataFrame
       confusion matrix
    """

    labels = list(set(answer))
    cm = confusion_matrix(answer, predict, labels)
    cm = pd.DataFrame(cm, columns=labels, index=labels)
    return cm


# 描画
def display_cm(answer, predict, title, fnum, time, save):
    """
    paramerters
    ---
    answer:numpy.ndarray (100)
           answer models
    predict:numpy.ndarray (100)
            predicted models
    title:str
          title name
    fnum:int
         file number
    time:float
         calculation time
    save:int(1 or else)
         save or not save
    """

    cm = calc_cm(answer, predict)
    acc = np.sum(answer == predict) / len(answer) * 100

    sns.heatmap(cm, annot=True, cbar=False, square=True, cmap="binary")
    plt.title(
        title
        + " data"
        + str(fnum)
        + "\nAccuracy: "
        + str(acc)
        + "%\nTime: "
        + str(time),
        fontsize=15,
    )
    plt.xlabel("predicted models", fontsize=15)
    plt.ylabel("answer models", fontsize=15)
    plt.tight_layout()
    if save == 1:
        plt.savefig("data" + str(fnum) + title + ".png")
    plt.show()


def main(args):
    fnum = args.fnum
    save = args.save
    method = args.method

    OUTPUT, A, PI, B, ANSWER = file(fnum)

    if method == "f":
        start = time.perf_counter()
        predict = forward(OUTPUT, A, PI, B)
        t = round(time.perf_counter() - start, 6)
        display_cm(ANSWER, predict, "Forward", fnum, t, save)

    elif method == "v":
        start = time.perf_counter()
        predict = viterbi(OUTPUT, A, PI, B)
        t = round(time.perf_counter() - start, 6)
        display_cm(ANSWER, predict, "Viterbi", fnum, t, save)

    else:
        print("error:" + str(method) + " is not function (f or v)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fnum", default=4)
    parser.add_argument("--method", default="v")
    parser.add_argument("--save", default=0)
    args = parser.parse_args()

    main(args)
