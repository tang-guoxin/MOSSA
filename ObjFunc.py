# coding = 'utf-8'
'''
test function
'''
import numpy as np


def f1(x):
    return -x[:, 0] ** 2 + x[:, 1]


def f2(x):
    return 0.5 * x[:, 0] + x[:, 1] + 1


def fun(x):
    n = x.shape[0]
    ret = np.zeros((n, 2))
    ret[:, 0] = f1(x)
    ret[:, 1] = f2(x)
    return ret
