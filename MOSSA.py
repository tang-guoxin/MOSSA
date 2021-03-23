# user/bin/env python3
# coding = 'utf-8'
####################################
'''
fun:    多目标目标函数，返回值应是一个多维矩阵
nVar:   种群数目
dim:    维度
lb:     搜索下边界
ub:     搜索上边界
vlb:    速度下边界
vub:    速度上边界
prod:   生产者比例
pred:   察觉危险者比例
max_iter: 最大迭代次数
eps:    误差上限
w:      动态权重区间
'''
####################################
import numpy as np


class creat():
    def __init__(self, fun, nVar, dim, lb, ub, vlb, vub, prod=0.2, pred=0.1, max_iter=100, eps=1e-6, w=None):
        if w is None:
            w = [0.1, 0.9]
        self.fun = fun
        self.nVar = nVar
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.vlb = vlb
        self.vub = vub
        self.prod = prod
        self.pred = pred
        self.max_iter = max_iter
        self.eps = eps
        self.w = w

        self.pop = np.random.uniform(lb, ub, (nVar, dim))
        self.vel = np.random.uniform(vlb, vub, (nVar, dim))
        pass

    def fit(self, strs):
        print(strs)


x = np.random.uniform(1, 3, (2, 2))
print(x)
