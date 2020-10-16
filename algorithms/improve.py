"""
Use Naive Bayes to optimize Hyperparameter.
"""


from algorithms.FSFOA import FSFOA
from sklearn.linear_model import BayesianRidge
from utils import get_CA
import numpy as np


def improve(X, Y, LSC, GSC):
    # Sampling， 27 sampling data
    Hyperparam, Acc = [], []
    for life_time in [5, 10, 20]:
        for area_limit in [20, 80, 200]:
            for transfer_rate in [0.01, 0.05, 0.1]:
                vector, _ = FSFOA(X, Y, LSC, life_time, area_limit, transfer_rate, GSC)
                CA = get_CA(vector, X, Y)
                Hyperparam.append([life_time, area_limit, transfer_rate])
                Acc.append(CA)
                print(life_time, area_limit, transfer_rate, CA)
    # print(Hyperparam, Acc)

    # Bayesian Regression
    model = BayesianRidge().fit(Hyperparam, Acc)
    Acc, param = 0, []
    for life_time in range(5, 21, 1):
        for area_limit in range(20, 201, 1):
            for transfer_rate in np.arange(0.01, 0.11, 0.01):
                Acc_hat = model.predict([[life_time, area_limit, transfer_rate]])[0]
                # print(life_time, area_limit, transfer_rate, Acc_hat)
                if Acc < Acc_hat:
                    Acc, param = Acc_hat, [life_time, area_limit, transfer_rate]
    return param

