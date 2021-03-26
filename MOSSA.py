# user/bin/env python3
# coding = 'utf-8'
####################################
Info = '''
---------------------------------------
create by T-XiaoMeng at 3/24/2021 18:32
v 1.0
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
rep_num:    存档空间大小
gdn:        网格细度
---------------------------------------
rep 是一个列表，长度是一个定值，每个元素也是一个列表，第一个值为位置坐标，第二个值为多目标函数的适应度值
---------------------------------------
'''
print(Info)
####################################
import numpy as np
import matplotlib.pyplot as plt

class creat():
    def __init__(self, fun, nVar, dim, num, lb, ub, vlb, vub, prod=0.2, pred=0.1, max_iter=100, eps=1e-6, w=None, rep_num=5, gdn=10):
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
        self.rep_num = rep_num
        pass

    def rep2vec(self, rep):
        col = rep[0][1].shape[0]
        ls = [[] for i in range(col)]
        for ary in rep:
            for j in range(col):
                ls[j].append(ary[1][j])
        return ls

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

    # 轮盘对赌选取 leader <- 将密度映射到 0-1 之间
    def choseLeader(self, rep, tho_oder):
        print(rep)
        print(tho_oder)
        all_tho = sum([thoi[1] for thoi in tho_oder])
        thos = [i[1] for i in tho_oder]
        thos.reverse()
        prob = [i/all_tho for i in np.cumsum(thos)]
        chose_val = [v[0] for v in tho_oder]
        print(chose_val)
        chose_dic = dict(zip(prob, chose_val))
        sel = np.random.random(1)

        print(chose_dic)
        return 0

        pass

    # 自适应网格优化: grid opt
    def gridOpt(self, rep):
        [Max, Min] = self.getBound(rep)
        dic, tho = {}, {}
        for ls in rep:
            idx = []
            for j in range(self.num):
                id = np.ceil(self.gdn * (ls[1][j] - Min[j]) / (Max[j] - Min[j]))
                idx.append(id)
            this_key = tuple(idx)
            if this_key in dic:
                dic[this_key].append(ls)
                tho[this_key] += 1
            else:
                dic[this_key] = [ls]
                tho[this_key] = 1
        tho_oder = sorted(tho.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
        sel = np.random.randint(tho_oder[0][1])
        if tho_oder[0][1]-1 == 0:
            dic.pop(tho_oder[0][0])
            del tho_oder[0]
        else:
            del dic[tho_oder[0][0]][sel]
            tho_oder[0] = (tho_oder[0][0], tho_oder[0][1]-1)
        new_rep = []
        for ks in dic.items():
            for l in ks[1]:
                new_rep.append(l)
        return new_rep, tho_oder
        pass


    # 初始化 rep 档案集: pareto 支配解
    def initRep(self, fval, pop):
        rep = []
        for i in range(self.nVar):
            flag = True
            for j in range(0, self.nVar):
                if np.all(fval[i, :] > fval[j, :]) == True:
                    flag = False
                    break
            if flag == True:
                rep.append([pop[i, :], fval[i, :]])
            if len(rep) > self.rep_num:
                [rep, tho_oder] = self.gridOpt(rep) # 是否可以等循环结束后再一次性网格优化?不可以，当需要消除多个点时，没有更好的算法进行取舍
        Leader = self.choseLeader(rep, tho_oder)
        print(len(rep))
        return rep
        pass

    def paretoDominant(self, pop, rep):
        pass

    def train(self, show=False):
        fval = self.fun(self.pop)
        rep = self.initRep(fval, self.pop)
        print('-----------------train-----------------')
        print(rep)
        Pa = self.rep2vec(rep)
        plt.plot(fval[:, 0], fval[:, 1], 'go')
        plt.hold
        plt.plot(Pa[0], Pa[1], 'rx')
        plt.show()
        pass


#%% -------------------------------------------
import ObjFunc
fun = ObjFunc.fun

mode = creat(fun, 200, 2, 2, 0, 1, -1, 1)

mode.train()

