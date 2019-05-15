import tensorflow as tf
import numpy as np
import os
import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow.python.data import Dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


california_housing_dataframe = pd.read_csv("https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv", sep=",")
# 加载数据

california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
# 随机化处理，通过数据的index对数据顺序重新随机排序

california_housing_dataframe["median_house_value"] /= 1000.0
# 对数据中的medianhousevalue这一列数据改成以千作为单位

# print(california_housing_dataframe.describe())
# 对数据有个初步了解


# ------------------------定义特征并配置特征列--------------------------
my_feature = california_housing_dataframe[["total_rooms"]]
feature_columns = [tf.feature_column.numeric_column("total_rooms")]
# 把数据中的totalrooms作为线性回归中的一个输入特征

# ------------------------定义目标--------------------------
targets = california_housing_dataframe["median_house_value"]
# 我们的目标是通过totalrooms预测medianhousevalue

# ------------------------配置线性回归模型--------------------------
my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
# 使用梯度下降法训练模型
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
# 应用梯度裁剪，确保梯度大小在训练期间不会变得过大
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)

# ------------------------定义输入函数--------------------------
# 将我们的数据输入到线性回归模型当中，输入函数主要是告诉tensorflow怎么对数据进行预处理
# 以及在训练期间如何进行批处理、随机处理、重复数据
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    features = {key: np.array(value) for key, value in dict(features).items()}
# 将pandas特征数据转换成numpy数组字典，我们之前定义的my_features，就会在这里被处理
# 然后，我们再把我们的数据，通过tensorflow的dataset api转换成tensorflow需要的dataset对象
# 如果我们一次性处理所有数据，做梯度下降的时候会很慢，所以选择了小批量梯度下降法，把原始数据按照周期数（num_epochs）分为一批批（batch）
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

# shuffle就是对数据进行随机处理，以便数据在训练期间以随机方式传递到模型
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
#构件一个迭代器，向LinearRegressor返回下一批数据
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

# ------------------------训练模型--------------------------
# 把特征和目标封装到my_input_fn，my_input_fn返回的features和labels作为输入
_ = linear_regressor.train(
    input_fn=lambda: my_input_fn(my_feature, targets),
    steps=100
)

# ------------------------评估模型--------------------------
prediction_input_fn = lambda:my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

predictions = linear_regressor.predict(input_fn=prediction_input_fn)

predictions = np.array([item['predictions'][0] for item in predictions])

mean_squared_error = metrics.mean_absolute_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)

print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)

min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print("min. median house value: %0.3f" % min_house_value)
print("max. median house value: %0.3f" % max_house_value)
print("difference between min and max: %0.3f" % min_max_difference)
print("root mean squared error: %0.3f" % root_mean_squared_error)

# 其实到这里，已经初步建立好模型了，只是这个模型的均方误差比较大，所以需要进一步改进
