import numpy as np
import math
from matplotlib import pyplot as plt

x = np.linspace(0, 4 * math.pi, 100)

# 拟合的函数是sin（x），加上随机误差
y = np.sin(x) + 0.1*np.random.randn(len(x))

# 拟合点x0
x0 = np.linspace(0, 13, 50)
yhatnw = np.zeros(len(x0))
n = len(x0)
h = 1

for i in range(n):
    # 通过normal kernel决定权重
    # [((x - x0[i])/h)**2]可以反映x的所有值到x0的一个拟合点的距离，以此距离根据核函数计算权重
    # w是x0的一个拟合点到x各点的权重的一维矩阵
    w = np.exp(-0.5*((x - x0[i])/h)**2) * (np.pi * 2)**(-1/2) * (h**-2)
    # 上面计算了x各点到拟合点x0的权重，然后这里利用权重和y，通过加权平均得到拟合点x0对应的y的拟合值
    # 简单来说，比如现在要估计x0的y值，nadarayaWatson做的就是分析其他样本点到x0的距离
    # 根据距离结合核函数确定权重，因为每个样本都有一个y值，根据各个y值和权重加权平均得到这个x0对应的y值
    # 加权平均是因为太远的点影响不大，故不考虑，kernel回归主要反映在权重通过kernel函数确定
    yhatnw[i] = sum(w * y) / sum(w)

plt.scatter(x, y)
plt.plot(x0, yhatnw, c='k')
plt.show()

