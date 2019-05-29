import pandas as pd
import numpy as np


# 生成训练数据，特征为xy坐标，标签为0，1（rank）
def loaddata():
    # 生成第一类别数据，主要分布在x轴的[0，1]区间内
    r1x = np.random.uniform(0, 1, 50)
    r1y = np.random.uniform(0, 1, 50)
    # 生成第一类别数据，主要分布在x轴的[1，2]区间内
    r2x = np.random.uniform(1, 2, 50)
    r2y = np.random.uniform(0, 1, 50)

    rawdata1 = {'x': r1x, 'y': r1y,  'rank': 0}
    rawdata2 = {'x': r2x, 'y': r2y,  'rank': 1}
    data1 = pd.DataFrame(rawdata1)
    data2 = pd.DataFrame(rawdata2)
    data = pd.concat([data1, data2], ignore_index=True)

    # 对合并数据随机排序
    data = data.sample(frac=1)

    return data, data1, data2


def knn(data, onetest, k):

    # 就算该测试样本与所有训练样本的欧氏距离
    data['distance'] = ((data['x'] - np.tile(onetest['x'], len(data['x'])))**2 + (data['y'] - np.tile(onetest['y'], len(data['x'])))**2)**0.5

    # 对距离排序，选择距离最小的k个训练样本数据
    data = data.sort_values(by='distance').head(k)

    # 计算这k个样本的不同标签的数量
    labels = data['rank'].value_counts()

    # 选择数量的标签作为测试样本的标签
    label = labels.head(1).index.values[0]
    return label


if __name__ == '__main__':

    # 读取训练样本
    data, data1, data2 = loaddata()

    # 生成测试样本
    testx = np.random.uniform(1, 2, 10)
    testy = np.random.uniform(0, 1, 10)
    test = {'x': testx, 'y': testy, 'rank': 0}
    test = pd.DataFrame(test)

    print('处理前的数据：')
    print(test)

    # 对测试样本逐个应用knn计算标签，这里选择k=4，可调整
    for i in range(test.shape[0]):
        label = knn(data, test[i: i+1], 4)
        test.loc[i, 'rank'] = label

    print('处理后的数据： ')
    print(test)
