import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 构建二维特征，标签为0，1（rank）
def loaddata():
    # 生成第一类别数据，主要分布在x轴的[0，1]区间内
    r1x = np.random.uniform(0, 1, 50)
    r1y = np.random.uniform(0, 1, 50)
    # 生成第一类别数据，主要分布在x轴的[1，2]区间内
    r2x = np.random.uniform(1, 2, 50)
    r2y = np.random.uniform(0, 1, 50)
    # 在标签内插入一列1，把截距整合到矩阵内
    con = np.ones(50)
    con = con.T

    rawdata1 = {'con': con, 'x': r1x, 'y': r1y,  'rank': 0}
    rawdata2 = {'con': con, 'x': r2x, 'y': r2y,  'rank': 1}
    data1 = pd.DataFrame(rawdata1)
    data2 = pd.DataFrame(rawdata2)
    data = pd.concat([data1, data2], ignore_index=True)

    # 对合并数据随机排序
    data = data.sample(frac=1)

    # 可视化数据点，用颜色代表不同类别
    # plt.scatter(data1['x'], data1['y'], c='r')
    # plt.scatter(data2['x'], data2['y'], c='b')
    # plt.show()

    return data, data1, data2


# 定义sigmoid函数
def sigmoid(z):
    a = 1/(1+np.exp(-z))
    return a


# 初始化theta为0
def initialize_with_zeros(x):
    theta = np.zeros([1, x.shape[1]])
    return theta


def logistic(theta, x, y, learning_rate=0.1, iter_max=10000):

    j = 0
    # 注意矩阵计算时矩阵形状要相互匹配，这是很常犯的错误
    y = y.T

    while j < iter_max:

        # 计算模型输出和损失函数
        yhat = sigmoid(np.dot(x, theta.T))

        # 这里选择的损失函数比较特别（具体见模型分析），我们要最大化损失函数（其实这里就是极大似然求参数的思想）
        # 所以和以前不一样，不能当损失函数小到一定程度就停止迭代
        # 当然我们也可以通过模型的准确率判断模型的优劣，但是没必要在这里复杂化参数迭代过程

        # 梯度上升更新theta
        theta = theta.T + learning_rate * x.T * (y - yhat)
        theta = theta.T

        j = j+1

    return theta


def predict(x, theta):
    # 这里主要是把模型的输出根据阈值决定最终的分类
    Y_prediction = []
    yhat = sigmoid(np.dot(x, theta.T))

    for i in range(x.shape[0]):
        # 阈值为0.5
        if yhat[i] > 0.5:
            Y_prediction.append(1)
        else:
            Y_prediction.append(0)

    Y_prediction = np.mat(Y_prediction)
    Y_prediction = Y_prediction.T

    return Y_prediction


def plot_model(theta):

    data, data1, data2 = loaddata()
    x = data[['con', 'x', 'y']]
    y = data['rank']
    classLabels = y
    dataMatIn = np.mat(x)

    n = np.shape(dataMatIn)[0]

    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(n):
        if classLabels[i] == 1:
            xcord1.append(dataMatIn[i, 1])
            ycord1.append(dataMatIn[i, 2])
        else:
            xcord2.append(dataMatIn[i, 1])
            ycord2.append(dataMatIn[i, 2])

    x = np.arange(0, 3, 0.1)
    y = (-theta[0, 0] - theta[0, 1] * x) / theta[0, 2]
    plt.plot(x, y)
    plt.scatter(data1['x'], data1['y'], c='r')
    plt.scatter(data2['x'], data2['y'], c='b')
    plt.axis([-0.5, 2.5, 0, 1])
    plt.show()


def logistic_model(x, y, learning_rate=0.1, iter_max=10000):

    theta = initialize_with_zeros(x)
    theta = logistic(theta, x, y, learning_rate, iter_max)

    prediction = predict(x, theta)
    y = y.T
    y = np.array(y)
    prediction = np.array(prediction)

    # 这里定义了模型的准确率，主要是模型能正确分类的数量占总数量的比率
    accuracy = 1 - sum(np.abs(y-prediction))/len(y)

    print('accuracy is ', float(accuracy))

    plot_model(theta)


if __name__ == '__main__':

    data, data1, data2 = loaddata()

    x = data[['con', 'x', 'y']]
    y = data['rank']
    y = np.mat(y)
    x = np.mat(x)

    logistic_model(x, y, learning_rate=0.1, iter_max=10000)
