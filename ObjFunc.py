# coding = 'utf-8'
'''
test function
'''

import numpy as np

#---------------------------------------------------------------------------

def f1(x):
    return -x[:, 0] ** 2 + x[:, 1]


def f2(x):
    return 0.5 * x[:, 0] + x[:, 1] + 1

def testfun1(x):
    n = x.shape[0]
    ret = np.zeros((n, 2))
    ret[:, 0] = f1(x)
    ret[:, 1] = f2(x)
    return ret

# ---------------------------------------------------------------------------

def tf21(x):
    x1, x2, x3 = np.power(x[:, 0], 2), np.power(x[:, 1], 2), np.power(x[:, 2], 2)
    return -10*np.exp(-0.2*np.sqrt(x1+x2)) - 10*np.exp(-0.2*np.sqrt(x2+x3))

def tf22(x):
    ret = 0
    for i in range(3):
        ret += np.power(np.abs(x[:, i]), 0.8) + 5*np.power(np.sin(x[:, i]), 3)
    return ret

def testfun2(x):
    n = x.shape[0]
    ret = np.zeros((n, 2))
    ret[:, 0] = tf21(x)
    ret[:, 1] = tf22(x)
    return ret
# --------------------------------------------------------------------------
def tf31(x):
    return x[:, 0]

def g3(x):
    return 11 + np.power(x[:, 1], 2) - 10 * np.cos(2*np.pi*x[:, 1])


def h3(x):
    n = x.shape[0]
    ret = np.zeros((n, 1))
    for i in range(n):
        if x[i, 0] <= 11 + x[i, 1]**2 - 10*np.cos(2*np.pi*x[i, 1]):
            ret[i, :] = 1 - np.sqrt(x[i, 0]/(11 + x[i, 1]**2 - 10 * np.cos(2*np.pi*x[i, 1])))
        else:
            continue
    return ret[:, 0]

def tf32(x):
    return g3(x) * h3(x)

def testfun3(x):
    n = x.shape[0]
    ret = np.zeros((n, 2))
    ret[:, 0] = tf31(x)
    ret[:, 1] = tf32(x)
    return ret
# ----------------------------------------------------------------------
def tf41(x):
    return x[:, 0]

def tf42(x):
    return g4(x[:, 1]) / x[:, 0]

def g4(x2):
    tmp1 = 2 - np.exp(-np.power((x2-0.2)/0.004, 2))
    tmp2 = -0.8 * np.exp(-np.power((x2-0.6)/0.4, 2))
    return tmp1 + tmp2

def testfun4(x):
    n = x.shape[0]
    ret = np.zeros((n, 2))
    ret[:, 0] = tf41(x)
    ret[:, 1] = tf42(x)
    return ret
# ----------------------------------------------------------------------
def x2f(x):
    n = x.shape[0]
    ret = np.zeros((n, 2))
    ret[:, 0] = np.power(x, 2)[:, 0]
    ret[:, 1] = np.power(x-1, 2)[:, 0]
    return ret
######################################################################
# x = np.random.uniform(0, 1, (5, 2))
#-----------------------------------------
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
# x = np.mat([0.88221571, 0.52562353])
# print(x2f(x))
# print(test3(x))

# print(testfun4(x))
