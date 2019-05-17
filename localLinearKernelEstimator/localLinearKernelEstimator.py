import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0, 4 * np.pi, 100)

# 拟合的函数是sin（x），加上随机误差
y = np.sin(x) + 0.75*np.random.randn(len(x))

# 这里没有定义拟合点，直接用样本点的x作为拟合点，为每个x拟合一个新的更平缓的y值

h = 1
yhatlin = np.zeros(len(x))
n = len(x)

# 对每一个样本点x，借助其他所有的样本去计算该点对应的yhat
for i in range(n):
    w = (2 * np.pi * h**2)**(-1/2) * np.exp(-0.5*((x - x[i])/h)**2)
    xc = x-x[i]
    s2 = sum(xc**2*w)/len(x)
    s1 = sum(xc*w)/len(x)
    s0 = sum(w)/len(x)
    yhatlin[i] = sum(((s2-s1*xc)*w*y) / (s2*s0-s1**2)) / n


plt.scatter(x, y)
plt.plot(x, yhatlin)
plt.show()
