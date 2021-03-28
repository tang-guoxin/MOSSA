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
abe:        变异率(仅在迭代的前80%发生)
---------------------------------------
rep 是一个列表，长度是一个定值，每个元素也是一个列表，第一个值为位置坐标，第二个值为多目标函数的适应度值
---------------------------------------
'''
print(Info)
####################################
import numpy as np
import matplotlib.pyplot as plt


class Creat():
    def __init__(self, fun, nVar, dim, num, lb, ub, vlb, vub, prod=0.2, pred=0.1, max_iter=100, w=None, rep_num=None, gdn=10, abe=0.05):
        if w is None:
            w = [0.1, 0.9]
        if rep_num is None:
            rep_num = nVar
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
        self.w = w
        self.gdn = gdn
        self.abe = abe
        # 初始化种群
        self.pop = np.random.uniform(lb, ub, (nVar, dim))
        self.his_pop = None
        # 初始化速度
        self.vel = np.random.uniform(vlb, vub, (nVar, dim))
        # 初始化存档个数
        self.rep_num = rep_num
        # 标记生产者
        self.label = [False for i in range(nVar)]
        pass

    # 取出存档的函数值
    def rep2vec(self, rep):
        col = rep[0][1].shape[0]
        ls = [[] for i in range(col)]
        for ary in rep:
            for j in range(col):
                ls[j].append(ary[1][j])
        return ls

    # 获取 rep 边界
    def getBorder(self, rep):
        Max, Min = -0xffffff, 0xffffff
        ary = []
        for ls in rep:
            ary.append(ls[1])
        arys = np.array(ary)
        Max = np.max(arys, 0) + 0.1
        Min = np.min(arys, 0) - 0.1
        return Max, Min

    # 轮盘对赌选取 leader <- 将密度映射到 0-1 之间 且网格密度越低, 被选取的概率越大
    def choseLeader(self, tho_oder, dic):
        all_tho = sum([thoi[1] for thoi in tho_oder])
        thos = [i[1] for i in tho_oder]
        thos.reverse()
        prob = [i / all_tho for i in np.cumsum(thos)]
        key_prob = [(0, prob[0])] + [(prob[i - 1], prob[i]) for i in range(1, len(prob))]
        chose_val = [v[0] for v in tho_oder]
        chose_dic = dict(zip(key_prob, chose_val))
        sel = np.random.random()
        for itv in key_prob:
            if sel > itv[0] and sel < itv[1]:
                leader = chose_dic[itv]
                sel_loc = np.random.randint(len(dic[leader]))
                return dic[leader][sel_loc][0]
        return False

    # 计算网格密度, 返回: tho_oder 是每个网格的密度 且已按照从大到小排序 dic 是每个网格对应的解和位置信息
    def calcTho(self, rep):
        [Max, Min] = self.getBorder(rep)
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
        tho_oder = sorted(tho.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        return tho_oder, dic

    # 自适应网格优化: grid opt 用来更新rep 当rep满了的时候 去除一些较差的Pareto解
    def gridOpt(self, rep):
        [tho_oder, dic] = self.calcTho(rep)
        sel = np.random.randint(tho_oder[0][1])
        if tho_oder[0][1] - 1 == 0:
            dic.pop(tho_oder[0][0])
            del tho_oder[0]
        else:
            del dic[tho_oder[0][0]][sel]
            tho_oder[0] = (tho_oder[0][0], tho_oder[0][1] - 1)
        new_rep = []
        for ks in dic.items():
            for l in ks[1]:
                new_rep.append(l)
        return new_rep

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
                rep = self.gridOpt(rep)  # 是否可以等循环结束后再一次性网格优化?不可以，当需要消除多个点时，没有更好的算法进行取舍
        [tho_oder, dic] = self.calcTho(rep)
        Leader = self.choseLeader(tho_oder, dic)
        return rep, Leader

    # 判断是否Pareto占优, 如果是, 则允许更新新的位置
    def isParetoDominant(self, pop_i, his_pop):
        # print(pop_i, ":", his_pop)
        # print(self.fun(np.mat(his_pop)), ':', self.fun(np.mat(pop_i)))
        if np.all(self.fun(np.mat(pop_i)) < self.fun(np.mat(his_pop))):
            return True
        else:
            return False
        return False

    # 处理第 i 个值的边界 vel or pop
    def dealBorder(self, i, name='vel'):
        if name == 'vel':
            vel_lb = np.where(self.vel[i, :] < self.vlb)[0].shape[0]
            vel_ub = np.where(self.vel[i, :] > self.vub)[0].shape[0]
            if vel_lb == 0:
                pass
            else:
                idx = np.where(self.vel[i, :] < self.vlb)
                self.vel[i, :][idx] = self.vlb
            if vel_ub == 0:
                pass
            else:
                idx = np.where(self.vel[i, :] > self.vub)
                self.vel[i, :][idx] = self.vub
            return True
        if name == 'pop':
            pop_lb = np.where(self.pop[i, :] < self.lb)[0].shape[0]
            pop_ub = np.where(self.pop[i, :] > self.lb)[0].shape[0]
            if pop_lb == 0:
                pass
            else:
                idx = np.where(self.pop[i, :] < self.lb)
                self.pop[i, :][idx] = self.lb
            if pop_ub == 0:
                pass
            else:
                idx = np.where(self.pop[i, :] > self.ub)
                self.pop[i, :][idx] = self.ub
            return True
        print('Error name = "pop" or "vel"...')
        return False

    # 选择身生产者
    def choseProd(self, leader):
        dis = {}
        self.label = [False for i in range(self.nVar)]
        for i in range(self.nVar):
            dis[i] = np.sum(np.power(leader - self.pop[i, :], 2))
        ls = sorted(dis.items(), key=lambda kv: (kv[1], kv[0]), reverse=False)
        for i in range(int(np.ceil(self.prod * self.max_iter))):
            self.label[ls[i][0]] = True
        return True

    # 更新生产者: 在MOSSA中, 我们认为, 离 Pareto 前端越靠近的点作为生产者
    def refProduct(self, ite, leader, rep, his_pop):
        for i in range(self.nVar):
            if self.label[i] == False:
                continue
            if np.random.random() > 0.5 + np.random.random()/5:
                self.pop[i, :] = self.pop[i, :] + np.random.randn()
                self.dealBorder(i, 'pop')
            else:
                self.vel[i, :] = np.random.random() * (his_pop[i, :] - self.pop[i, :]) + np.random.random() * (leader - self.pop[i, :])
                self.dealBorder(i, 'vel')
                new_w = self.w[1] - ite * (self.w[1] - self.w[0]) / self.max_iter
                self.pop[i, :] = self.pop[i, :] + new_w * self.vel[i, :]
                self.dealBorder(i, 'pop')
            if self.isParetoDominant(self.pop[i, :], his_pop[i, :]) == True:
                his_pop[i, :] = self.pop[i, :]
                print('新的Pareto.')
                fval = self.fun(np.mat(self.pop[i, :]))
                fval = np.array([fval[0][i] for i in range(self.num)])
                rep.append([self.pop[i, :], fval])
                if len(rep) > self.rep_num:
                    rep = self.gridOpt(rep)
        return rep, his_pop

    # 更新跟随者
    def refFollow(self, leader, rep, his_pop):
        A = np.random.randint(0, 2, self.dim)
        A[np.where(A == 0)] = -1
        A = np.mat(A)
        Ap = A.T * np.linalg.inv(A*A.T)
        ls = [Ap[i, 0] for i in range(self.dim)]
        Ap = np.array(ls)
        for i in range(self.nVar):
            if self.label[i] == True:
                continue
            self.pop[i, :] = leader + np.abs(self.pop[i, :]-leader) @ Ap
            self.dealBorder(i, 'pop')
            if self.isParetoDominant(self.pop[i, :], his_pop[i, :]) == True:
                his_pop[i, :] = self.pop[i, :]
                print('新的Pareto.')
                fval = self.fun(np.mat(self.pop[i, :]))
                fval = np.array([fval[0][i] for i in range(self.num)])
                rep.append([self.pop[i, :], fval])
                if len(rep) > self.rep_num:
                    rep = self.gridOpt(rep)
        return rep, his_pop

    # 更新察觉危险者
    def refPerceivedRisk(self, leader, rep, his_pop):
        idx = [i for i in range(self.nVar)]
        np.random.shuffle(idx)
        for i in range(int(self.pred*self.nVar)):
            self.pop[idx[i], :] = leader + np.random.randn() * np.abs(self.pop[idx[i], :]-leader)
            self.dealBorder(idx[i], 'pop')
            if self.isParetoDominant(self.pop[idx[i], :], his_pop[idx[i], :]) == True:
                his_pop[idx[i], :] = self.pop[idx[i], :]
                print('新的Pareto.')
                fval = self.fun(np.mat(self.pop[idx[i], :]))
                fval = np.array([fval[0][i] for i in range(self.num)])
                rep.append([self.pop[idx[i], :], fval])
                if len(rep) > self.rep_num:
                    rep = self.gridOpt(rep)
        return rep, his_pop

    # 引入变异
    def variation(self):
        for i in range(self.nVar):
            if np.random.random() < self.abe:
                self.pop[i, :] = np.random.uniform(self.lb, self.ub, (1, self.dim))
        return True

    # 初始化 his_pop
    def initHis(self):
        his_pop = np.ones_like(self.pop)
        for i in range(self.nVar):
            his_pop[i, :] = self.pop[i, :]
        return his_pop

    # 处理最终的 rep 集合
    def generatePareto(self, rep):
        new_rep = []
        for rp1 in rep:
            flag = True
            for rp2 in rep:
                if np.all(rp1[1] > rp2[1]):
                    flag = False
            if flag == True:
                new_rep.append(rp1)
        return new_rep

    # 开始搜索
    def train(self, show=False):
        fval = self.fun(self.pop)
        [rep, leader] = self.initRep(fval, self.pop)
        his_pop = self.initHis()
        for ite in range(self.max_iter):
            print('第{}次搜索,当前rep数量{}.\n'.format(ite+1, len(rep)))
            if ite < 0.8*self.max_iter:
                self.variation()
            self.choseProd(leader)
            [rep, his_pop] = self.refProduct(ite, leader, rep, his_pop)
            [rep, his_pop] = self.refFollow(leader, rep, his_pop)
            [rep, his_pop] = self.refPerceivedRisk(leader, rep, his_pop)
            fval = self.fun(self.pop)
            rep = self.generatePareto(rep)
            [tho_oder, dic] = self.calcTho(rep)
            leader = self.choseLeader(tho_oder, dic)
        rep = self.generatePareto(rep)
        Pa = self.rep2vec(rep)
        plt.subplot(1, 2, 1)
        plt.plot(fval[:, 0], fval[:, 1], 'go')
        plt.subplot(1, 2, 2)
        plt.plot(Pa[0], Pa[1], 'ro')
        plt.show()
        pass


# %% -------------------------------------------
import ObjFunc

ZDT1 = ObjFunc.ZDT1

fun1 = ObjFunc.fun

mode = Creat(fun=ZDT1, nVar=200, dim=30, num=2, lb=0, ub=1, vlb=-0.5, vub=0.5,
             prod=0.2, pred=0.1, max_iter=100, w=None, rep_num=1000, gdn=20, abe=0.5)

mode.train()
