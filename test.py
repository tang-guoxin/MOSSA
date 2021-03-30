import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig, ax = plt.subplots()
# 生成子图，相当于fig = plt.figure(),
# ax = fig.add_subplot(),其中ax的函数参数表示把当前画布进行分割，
# 例：fig.add_subplot(2,2,2).表示将画布分割为两行两列，ax在第2个子图中绘制，其中行优先。
x = np.arange(0, 2*np.pi, 0.01)  # 表示从0~2*np.pi之间每隔0.01取一个点
line, = ax.plot(x, np.sin(x))  # 注意，这里line后面要加上逗号，表示一个具有一个元素的元组


# print(type(line))
# print(type((line,)))
# <class 'matplotlib.lines.Line2D'>
# <class 'tuple'>

def animate(i):  # 这里的i其实就是参数0-99，即时frames控制的参数，控制程序画图变换的次数
    # print(i)  # 0-99
    line.set_ydata(np.sin(x + i/10.0))  # 改变线条y的坐标值
    return line,


def init():  # 初始化函数，图形开始显示的状态
    line.set_ydata(np.sin(x))
    return line,


ani = animation.FuncAnimation(fig=fig, func=animate, frames=100, init_func=init,interval=20, blit=False)
"""frames设定帧数,总共执行100个update就会进行下一次循环，并且frames还会作为参数传入animate()函数，init_func设定初始函数图像,
interval设置更新间隔此处设置为20毫秒，(仔细想想20毫秒其实是很小的一个间隔)
blit如果是只有变化了的像素点才更新就设置为True,如果是整张图片所有像素点全部更新的话就设置为False
"""
plt.show()
