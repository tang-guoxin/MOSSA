# coding = 'utf-8'

import numpy as np
import matplotlib.pyplot as plt
import ObjFunc as oj



def MakeParetoFront(lb, ub, fun, dim):
    x = np.zeros((100, dim))
    for i in range(dim):
        x[:, i] = np.linspace(lb[i], ub[i], 100)
    fval = fun(x)
    xx = fval[:, 0]
    yy = fval[:, 1]
    plt.plot(xx, yy, 'rx')
    plt.show()
    pareto = [[], []]
    for i in range(100):
        flag = True
        for j in range(100):
            if xx[i] > xx[j] and yy[i] > yy[j]:
                flag = False
        if flag == True:
            pareto[0].append(xx[i])
            pareto[1].append(yy[i])
    return pareto

#
# fun = oj.testfun3
#
# pareto = MakeParetoFront((0, -30), (1, 30), fun, 2)
# print(pareto)
# plt.plot(pareto[0], pareto[1], 'go')
# plt.show()
