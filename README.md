## 关于本项目

机器学习最难的应该是入门，怎么开始学习、机器学习到底是什么，网上虽然有很多资源，但是总感觉不知道从哪里入手。本项目主要是记录我自己学习时的一些心得体会，包括我认为的一个较好的机器学习的学习顺序，以及一些机器学习模型的代码。（想浏览更多文章也可访问我的个人网站：https://jesseyule.github.io/ ）

关于模型，我尽可能通俗地讲解，我觉得先有一个感性和整体的认识对后续深入的学习是有利的。关于代码，我是用python写模型，并力求做到简单易懂，甚至不需要读取数据文件，直接从代码中生成数据，这样的目的是希望当你下载代码之后，只需要按一下运行就能得到结果，根据结果再慢慢研究代码细节，我觉得这样能更有动力去学习。

因为本人也是刚入门的新手，所以在很多地方可能理解有误，特别是代码可能写得不够规范，我会在后续更新中不断修改，如果发现什么问题疑问，也欢迎交流。



### 前期准备

* 基础数学知识（矩阵、梯度、偏导数等，也可遇到这些概念再去了解学习）

* 掌握python基础（我用的IDE是pycharm，建议同时安装anaconda，安装完成即可拥有大量常用的数据科学python库）

* 了解Pandas、Numpy、Matplotlib这三个python库

  

### 学习路线

首先我推荐学习谷歌的一个速成课程，对机器学习有一个简单基础的认识：

- [机器学习速成课程](https://developers.google.com/machine-learning/crash-course/?hl=zh-cn)

另一方面，我自己也写了一篇快速入门：

- [机器学习概述](https://jesseyule.github.io/ai/introduction/content.html)

关于机器学习中的一些比较通用的过程、方法，我也写了一些专题讲解：

- [梯度下降法](https://jesseyule.github.io/ai/gradientDescent/content.html)
- [特征工程](https://jesseyule.github.io/ai/featureEngineering/content.html)
- [正则化与特征缩减](https://jesseyule.github.io/ai/regularization/content.html)
- [特征降维](https://jesseyule.github.io/ai/dimensionReduction/content.html)
- [特征子集选择](https://jesseyule.github.io/ai/subsetSelection/content.html)
- [交叉检验](https://jesseyule.github.io/ai/crossValidation/content.html)
- [机器学习模型评价指标](https://jesseyule.github.io/ai/modelEvaluate/content.html)

通过上面的过程，我认为基本上就可以了解机器学习做的是什么，还有具体的过程是怎么进行的了，接下来，就要从实际的模型入手，慢慢讲解这些模型是怎么"学习经验"的了，机器学习主要有两个方向：回归和分类，所以我也是从分别从简单到复杂，逐步介绍相关的一些模型算法。

### 回归模型

- [线性回归与多项式回归](https://jesseyule.github.io/ai/linearRegression/content.html)
- [LOESS](https://jesseyule.github.io/ai/loess/content.html)
- [核回归](https://jesseyule.github.io/ai/kernelRegression/content.html)
- [回归样条](https://jesseyule.github.io/ai/regressionSplines/content.html)
- [分位数回归](https://jesseyule.github.io/ai/quantileRegression/content.html)

简单介绍一下这些模型，线性回归主要用于线性分布的数据，可是现实中很少会有线性分布的数据，多项式回归就是为了解决非线性问题。可是多项式回归容易过拟合，针对训练数据的整体进行拟合也会导致一个训练数据的改变会影响整个模型，所以有了LOESS、核回归、回归样条这些先针对局部数据进行拟合，再把局部拟合结果组合成整体的算法。

回归分析的本质就是一个条件期望函数，有时候我们不仅仅希望分析响应变量的条件期望，还希望了解它的分布状况，所以就需要分位数回归，分位数回归不是单独一种回归模型，它是一种思想，只需要把损失函数的最小二乘法改成加权最小二乘法，就可以应用到线性回归、多项式回归等模型里面。

### 分类模型

- [KNN](https://jesseyule.github.io/ai/knn/content.html)
- [logistic回归](https://jesseyule.github.io/ai/logisticRegression/content.html)

KNN是最容易理解的分类模型，甚至也是最容易的机器学习模型，可以作为机器学习的开端。logistic回归其实就是线性模型应用到分类问题的一个例子。

### 数据生成

最后还想介绍一下数据生成的原理，因为计算机一般来说只能生成均匀分布的随机数，为了生成服从特定分布的数据我们就了解概率转换和蒙特卡罗方法。

- [概率分布的转换](https://jesseyule.github.io/math/transformation/content.html)
- [蒙特卡罗方法](https://jesseyule.github.io/ai/monteCarloMethod/content.html)