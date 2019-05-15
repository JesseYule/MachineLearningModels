import numpy as np
import matplotlib.pyplot as plt

# 定义要预测的函数，根据这个函数生成一些输入数据，并添加服从正态分布的随机误差项
x = 10*np.random.random((1, 100))
epsilon = np.random.normal(0, 1, 100)
z = np.zeros((1, 100))
y = 5*x + 11 + epsilon

# 初始化要预测的参数
a = 3
b = 6

# 定义损失函数，这里用RSE，计算梯度
RSE = 0
gra_a = 0
gra_b = 0

# 最大迭代次数
iter_max = 10000
j = 0

while j < iter_max:

    # 计算RSE
    for i in range(len(x[0, :])):
        RSE = RSE + ((y[0, i] - a * x[0, i] - b) ** 2)/len((x[0, :]))

    # 计算ab的梯度
    for i in range(len(x[0, :])):
        gra_a = gra_a + (-2*y[0, i]*x[0, i] + 2*a*x[0, i]**2 + 2*x[0, i]*b)/len((x[0, :]))
        gra_b = gra_b + (-2*y[0, i] + 2*a*x[0, i] + 2*b)/len((x[0, :]))

    # 当RSE小于1，停止迭代
    if RSE < 1:
        print(['the result is :', a, b])
        break
    else:
        a = a - 0.001*gra_a
        b = b - 0.001*gra_b

    # 重置参数
    j = j+1
    gra_a = 0
    gra_b = 0
    RSE = 0

# 生成符合拟合函数的点画线
xhat = np.arange(0, 10, 0.1)
yhat = a*xhat+b


plt.scatter(x, y)
plt.plot(xhat, yhat, c='r')
plt.show()





