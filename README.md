## 关于本项目

学习机器学习最难的应该是入门的那一步，怎么开始学习、机器学习到底是什么，网上虽然有很多资源，但是总感觉不知道从哪里入手。本项目主要是记录我自己学习的一些心得体会，包括我认为的一个学习机器学习的顺序，以及一些机器学习的模型的代码。

关于模型，我尽可能通俗地讲解，我觉得先有一个感性和整体的认识对后续深入的学习是有利的。关于代码，我用python写模型，并力求做到简洁，甚至不需要读取数据文件，直接从代码中生成数据，这样的目的是希望当你下载代码之后，只需要按一下运行就能得到结果，根据结果再慢慢研究代码细节，我觉得这样能更有动力去学习。

因为本人也是刚入门的新手，所以在很多地方可能理解有误，代码也可能写得不够标准，如果发现什么问题疑问，欢迎交流。



### 前期准备

* 基础数学知识（矩阵、梯度、偏导数等，也可遇到这些概念再去了解学习）

* 掌握python基础

* 了解Pandas、Numpy、Matplotlib这三个python库

  

### 学习路线

首先我推荐学习谷歌的一个速成课程，对机器学习有一个简单基础的认识：

- [机器学期速成课程](https://developers.google.com/machine-learning/crash-course/?hl=zh-cn)

另一方面，我自己也写了一篇快速入门：

- [机器学期概述](https://jesseyule.github.io/ai/introduction/content.html)

关于机器学习中的一些过程，我也写了一些专题讲解：

- [梯度下降法](https://jesseyule.github.io/ai/gradientDescent/content.html)
- [特征工程](https://jesseyule.github.io/ai/featureEngineering/content.html)
- [正则化与特征缩减](https://jesseyule.github.io/ai/regularization/content.html)
- [特征降维](https://jesseyule.github.io/ai/dimensionReduction/content.html)
- [特征子集选择](https://jesseyule.github.io/ai/subsetSelection/content.html)
- [交叉检验](https://jesseyule.github.io/ai/crossValidation/content.html)
- [机器学习模型评价指标](https://jesseyule.github.io/ai/modelEvaluate/content.html)

通过上面的过程，我认为基本上就可以了解机器学习做的是什么，还有具体的过程是怎么进行的了，接下来，就要从实际的模型入手，慢慢讲解怎么通过模型进行学习，机器学习主要有两类的方向：回归和分类，所以我也是从分别从简单到复杂，逐步介绍相关的一些模型算法。

### 回归模型

- [线性回归与多项式回归](https://jesseyule.github.io/ai/linearRegression/content.html)
- [LOESS](https://jesseyule.github.io/ai/loess/content.html)
- [核回归](https://jesseyule.github.io/ai/kernelRegression/content.html)
- [回归样条](https://jesseyule.github.io/ai/regressionSplines/content.html)

简单介绍一下这些模型，线性回归主要用于线性分布的数据，可是现实中很少会有线性分布的数据，多项式回归就是为了解决非线性问题。可是多项式回归容易过拟合，针对训练数据的整体进行拟合也会导致一个训练数据的改变会影响整个模型，所以有了LOESS、核回归、回归样条这些针对局部的数据进行拟合，最后整合成整体的算法。

### 分类模型

- [logistic回归](https://jesseyule.github.io/ai/logisticRegression/content.html)
- [KNN](https://jesseyule.github.io/ai/knn/content.html)

logistic回归其实就是线性模型应用到分类问题的一个例子，KNN则是最容易理解的分类模型。



