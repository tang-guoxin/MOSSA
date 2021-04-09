#c coding = 'utf-8'

import ObjFunc
import MOSSA
import cmper
import matplotlib.pyplot as plt

# fx2 = ObjFunc.x2f
#
# test1 = ObjFunc.testfun1 # ---
# test2 = ObjFunc.testfun2 # [-5, 5] dim = 3
# test3 = ObjFunc.testfun3 # [0, -30] [1, 30] dim = 2
# test4 = ObjFunc.testfun4 # [0.1, 1] [0.1, 1] dim = 2
# test5 = ObjFunc.testfun5 # dim = 4
# --------------------------------------------------------------------

[fun, nVar, dim, lb, ub] = ObjFunc.choseFun('test2')
isTest1 = False
# --------------------------------------------------------------------
# 超参数: 8 个
mode = MOSSA.Creat(fun=fun, nVar=nVar, dim=dim, num=2, lb=lb, ub=ub, vlb=-1, vub=1,
             prod=0.2, pred=0.1, max_iter=100, w=None, rep_num=100, gdn=7, abe=0.1, isTest1=isTest1)

[x1, y1] = mode.train(show=True)
[x2, y2] = cmper.main(display=True)

plt.plot(x1, y1, 'ro')
plt.hold
plt.plot(x2, y2, 'gx')
plt.show()
