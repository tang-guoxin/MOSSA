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

def ZDT1g(x):
    n = x.shape[1]
    return 1 + 9/(n-1) * np.sum(x[:, 1:n], 1)
def ZDT1h(f, g):
    return 1 - np.sqrt(f/g)

def ZDT1(x):
    n = x.shape[0]
    ret = np.zeros((n, 2))
    ret[:, 0] = x[:, 0]
    ret[:, 1] = ZDT1g(x) * ZDT1h(ret[:, 0], ZDT1g(x))
    return ret


######################################################################
# x = np.random.uniform(0, 1, (55, 2))
# y = ZDT1(x)
# print(y)
#
# import matplotlib.pyplot as plt
#
# plt.plot(y[:, 0], y[:, 1], 'ro')
# plt.show()
#
# x2 = np.mat([0.44187983, 0.75345132])
# print(ZDT1(x2))
