# user/bin/env python3
# coding = 'utf-8'
####################################
'''
fun:    多目标目标函数，返回值应是一个多维矩阵
nVar:   种群数目
dim:    维度
num:    函数个数
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
    def __init__(self, fun, nVar, dim, num, lb, ub, vlb, vub, prod=0.2, pred=0.1, max_iter=100, eps=1e-6, w=None, arc=0.2):
        if w is None:
            w = [0.1, 0.9]
        self.fun = fun
        self.nVar = nVar
        self.dim = dim
        self.num = num
        self.lb = lb
        self.ub = ub
        self.vlb = vlb
        self.vub = vub
        self.prod = prod
        self.pred = pred
        self.max_iter = max_iter
        self.eps = eps
        self.w = w
        # 初始化种群
        self.pop = np.random.uniform(lb, ub, (nVar, dim))
        # 初始化速度
        self.vel = np.random.uniform(vlb, vub, (nVar, dim))
        # 初始化存档
        # self.rep = np.ones((np.floor(arc*nVar), dim))
        self.rep_num = np.floor(arc*nVar)
        pass

    def paretoDominant(self, pop, rep):
        n = len(rep)
        for i in range(self.nVar):
            for j in range(n):
                for k in range(self.num):
                    if pop[i, k] > pop[j][k]
        pass

    def fit(self, strs):
        print(strs)


x = np.random.uniform(1, 3, (2, 2))
print(x)
