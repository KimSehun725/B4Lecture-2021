# include flake8, black

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main(args):
    # fname = args.fname
    fname = "data1.csv"

    # get current working directory
    path = os.path.dirname(os.path.abspath(__file__))

    # For example, if fname = data1.csv, graphtitle = data1
    graphtitle, _ = os.path.splitext(fname)
    save_fname = fname.replace("csv", "png")
    fname = os.path.join(path, "data", fname)
    save_fname = os.path.join(path, "result", save_fname)

    # load csv file and convert to ndarray
    data = pd.read_csv(fname, header=None).values
    df = pd.DataFrame(data)
    print(df.info())

    if data.shape[1] == 1:
        x = data[:, 0]
        y = [0] * data.shape[0]

        fig = plt.figure(figsize=(6.4, 2.4))
        ax = fig.add_subplot(111, title=graphtitle, xlabel="X", yticks=[0])
        ax.tick_params(labelleft=False, left=False)

        # ax.scatter(x, y, s=20, c="darkblue")
        ax.scatter(
            x,
            y,
            s=20,
            linewidths=1.0,
            marker="o",
            facecolor="None",
            edgecolors="darkblue",
        )
        plt.tight_layout()
        plt.grid(ls=":")
        plt.savefig(save_fname, transparent=True)
        plt.show()

    if data.shape[1] == 2:
        x = data[:, 0]
        y = data[:, 1]

        fig = plt.figure()
        ax = fig.add_subplot(111, title=graphtitle, xlabel="X", ylabel="Y")
        ax.scatter(
            x,
            y,
            s=20,
            linewidths=1.0,
            marker="o",
            facecolor="None",
            edgecolors="darkblue",
        )
        plt.grid(ls=":")
        plt.savefig(save_fname, transparent=True)
        plt.show()


if __name__ == "__main__":
    # process args
    parser = argparse.ArgumentParser(description="Regression and Regularization.")
    """
    parser.add_argument("fname", type=str, help="Load Filename")
    """
    args = parser.parse_args()
    main(args)
