# user/bin/env python3
# coding = 'utf-8'
####################################
Version = '''
---------------------------------------
create by T-XiaoMeng at 3/24/2021 18:32
v 1.0.0
---------------------------------------
fun:        多目标目标函数，返回值应是一个多维矩阵
nVar:       种群数目
dim:        维度
num:        函数个数
lb:         搜索下边界
ub:         搜索上边界
vlb:        速度下边界
vub:        速度上边界
prod:       生产者比例
pred:       察觉危险者比例
max_iter:   最大迭代次数
eps:        误差上限
w:          动态权重区间
gdn:        网格细度
---------------------------------------
rep 是一个列表，每个元素也是一个列表，第一个值为位置坐标，第二个值为多目标函数的适应度值，长度是一个定值
'''
####################################
import numpy as np
import matplotlib.pyplot as plt

class creat():
    def __init__(self, fun, nVar, dim, num, lb, ub, vlb, vub, prod=0.2, pred=0.1, max_iter=100, eps=1e-6, w=None, arc=0.2, gdn=10):
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
        self.gdn = gdn
        # 初始化种群
        self.pop = np.random.uniform(lb, ub, (nVar, dim))
        # 初始化速度
        self.vel = np.random.uniform(vlb, vub, (nVar, dim))
        # 初始化存档
        # self.rep = np.ones((np.floor(arc*nVar), dim))
        self.rep_num = np.floor(arc*nVar)
        pass


    # 获取 rep 边界
    def getBound(self, rep):
        Max, Min = -0xffffff, 0xffffff
        ary = []
        for ls in rep:
            ary.append(ls[1])
        arys = np.array(ary)
        Max = np.max(arys, 0) + 0.1
        Min = np.min(arys, 0) - 0.1
        return Max, Min

    # 自适应网格优化: grid opt
    def gridOpt(self, rep):
        [Max, Min] = self.getBound(rep)
        print([Max, Min])
        pass


    # 初始化 rep 档案集: pareto 支配解
    def initRep(self, fval, pop):
        rep = []
        for i in range(self.nVar):
            flag = True
            for j in range(i, self.nVar):
                if np.all(fval[i, :]) > np.all(fval[j, :]):
                    flag = False
                    break
            if flag == True:
                rep.append([pop[i, :], fval[i, :]])
        if len(rep) > self.rep_num:
            rep = self.gridOpt(rep) # 是否可以等循环结束后再一次性网格优化?
        return rep
        pass

    def paretoDominant(self, pop, rep):
        pass

    def train(self, show=False):
        fval = self.fun(self.pop)
        rep = self.initRep(fval, self.pop)

        pass

import ObjFunc
fun = ObjFunc.fun
#%% -------------------------------------------
mode = creat(fun, 10, 2, 2, 0, 1, -1, 1)

mode.train()

