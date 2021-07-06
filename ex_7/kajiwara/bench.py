from pathlib import Path
from datetime import datetime

import pandas as pd

from gmm import em_algorithm
from utils import set_initial

TIME_TEMPLATE = '%Y%m%d%H%M%S'


def main():
    result_path = Path('./bench')
    timestamp = datetime.now().strftime(TIME_TEMPLATE)
    result_path = result_path/timestamp
    if not result_path.exists():
        result_path.mkdir(parents=True)

    # loading data
    df1 = pd.read_csv('../data1.csv', header=None)
    # df2 = pd.read_csv('../data2.csv', header=None)
    # df3 = pd.read_csv('../data3.csv', header=None)

    # df to nd ndarray
    data1 = df1.values
    # data2 = df2.values
    # data3 = df3.values

    ex_ite = 10
    K = 2
    epsilon = 0.00001

    params = ['kmeans', 'random']
    res = {}

    for p in params:
        sum_ite = 0
        for _ in range(ex_ite):
            Mu, Sigma, Pi = set_initial(data1, K, mean_type=p)
            ite, _, _, _, _, _, _ = em_algorithm(data1, Pi, Mu, Sigma, epsilon, output=False)
            sum_ite += ite
        res[p] = sum_ite / ex_ite

    print(res)


if __name__ == '__main__':
    main()
