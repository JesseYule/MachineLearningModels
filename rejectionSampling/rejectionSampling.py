import numpy as np
import math
import matplotlib.pyplot as plt

#假设我们有一个概率密度函数：0.3*0.5*exp(-0.5*x)+0.7*30*x^4*(1-x)
#通过rejection sampling生成样本

x = 10*np.random.random((1, 10000))
y = np.random.random((1, 10000))
i = 0
k = 0
observation = np.zeros((500, 1))

while k < 500:
    if 0.3*0.5*math.exp(-0.5*x[0, i])+0.7*30*x[0, i]**4*(1-x[0, i]) > y[0, i]:
        observation[k, 0] = x[0, i]
        k = k + 1

    i = i+1

plt.hist(observation, bins=10, facecolor="darkblue", edgecolor="black", alpha=0.5)
plt.show()
