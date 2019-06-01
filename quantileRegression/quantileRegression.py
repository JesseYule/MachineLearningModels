import numpy as np
import matplotlib.pyplot as plt


def loaddata():
    x = np.random.uniform(0, 1, 1000)
    y = np.random.uniform(0, 1, 1000)
    return x, y


def init():
    a = 1
    b = 1
    return a, b


def quantile(x, y, a, b, itermax=1000, learning_rate=0.001):

    # 假设目标函数形式是y = b
    MSE = 0
    QMSE = 0
    j = 0
    gra_a = 0
    gra_b = 0

    # 0.7分位数回归
    seventy = np.percentile(y, 70)
    q = 0.7

    while j < itermax:

        # 最小二乘法的损失函数
        for i in range(len(x)):
            MSE = MSE + ((y[i] - a) ** 2)/len(x)

        # 加权最小二乘法的损失函数
        for i in range(len(x)):
            if y[i] >= seventy:
                QMSE = QMSE + q*((y[i] - b) ** 2)/len(x)
            else:
                QMSE = QMSE + (1-q)*((y[i] - b) ** 2) / len(x)

        # 计算a、b的梯度
        for i in range(len(x)):
            gra_a = gra_a + (-2*y[i]+2*a)/len(x)
            if y[i] >= seventy:
                gra_b = gra_b + 0.7*(-2*y[i]+2*b)/len(x)
            else:
                gra_b = gra_b + 0.3*(-2*y[i]+2*b)/len(x)

        # 当MSE和QMSE都小于0.1，停止迭代
        if MSE < 0.1 and QMSE < 0.1:
            print(['the result is :', a, b])
            break
        else:
            a = a - learning_rate*gra_a
            b = b - learning_rate*gra_b

        j = j + 1
        gra_a = 0
        gra_b = 0
        MSE = 0
        QMSE = 0

    return a, b


x, y = loaddata()
a, b = init()

a, b = quantile(x, y, a, b, itermax=1000, learning_rate=0.001)

plt.scatter(x, y, s=1)

x = [0, 1]
ya = [a, a]
yb = [b, b]

plt.plot(x, ya, c='red')
plt.plot(x, yb, c='green')

plt.show()






