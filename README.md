# 鸢尾花分类项目 (Iris Classification)

## 项目概述

本项目实现了对著名的鸢尾花数据集(Iris Dataset)的分类算法。鸢尾花数据集包含三种不同类型的鸢尾花（山鸢尾、变色鸢尾和维吉尼亚鸢尾）的测量数据，每种花有50个样本，共150个样本。每个样本包含四个特征：花萼长度、花萼宽度、花瓣长度和花瓣宽度。

本项目实现了两种不同的分类算法：
1. K近邻算法 (KNN)
2. 支持向量机 (SVM)

对于每种算法，我们提供了自己实现的版本和使用sklearn库的版本，以便比较和学习。

## 项目结构

```
/iris_classify
├── data/                  # 数据文件夹
│   ├── iris.data          # 鸢尾花数据集
│   ├── bezdekIris.data    # 备份数据集
│   └── iris.names         # 数据集说明文件
├── models/                # 模型实现
│   ├── knn_classify_iris.py       # KNN算法实现
│   ├── knn_classify_iris2.0.py    # 改进版KNN算法实现
│   ├── knn(sklearn)_classify_iris.py  # 使用sklearn的KNN实现
│   ├── svm_classify_iris.py       # SVM算法实现
│   └── svm(sklearn)_classify_iris.py  # 使用sklearn的SVM实现
├── utils/                 # 工具函数
│   ├── data_loader.py     # 数据加载和预处理工具
│   └── metrics.py         # 评估指标和距离计算工具
├── visualization/         # 可视化工具
│   ├── plot_iris.py       # 数据可视化脚本
│   ├── visualize.py       # 可视化工具函数
│   └── img.png            # 可视化结果图
└── README.md              # 项目说明文件
```

## 算法实现

### KNN算法

我们实现了两个版本的KNN算法：

1. **基础版KNN算法** (`knn_classify_iris.py`)：
   - 实现了基本的K近邻算法
   - 使用欧几里得距离计算样本间距离
   - 通过投票方式确定类别

2. **改进版KNN算法** (`knn_classify_iris2.0.py`)：
   - 改进了数据划分方式
   - 使用加权KNN算法，距离越近的样本权重越大
   - 准确率显著提高，达到91%-97%

3. **使用sklearn的KNN实现** (`knn(sklearn)_classify_iris.py`)：
   - 使用sklearn库的KNeighborsClassifier
   - 作为参考和基准

### SVM算法

我们也实现了两个版本的SVM算法：

1. **自实现SVM算法** (`svm_classify_iris.py`)：
   - 实现了基本的支持向量机算法
   - 使用梯度下降优化参数

2. **使用sklearn的SVM实现** (`svm(sklearn)_classify_iris.py`)：
   - 使用sklearn库的SVC类
   - 使用RBF核函数
   - 作为参考和基准

## 数据可视化

项目包含数据可视化工具，可以绘制：
- 花萼特征（长度vs宽度）散点图
- 花瓣特征（长度vs宽度）散点图
- 混淆矩阵
- 决策边界

## 使用方法

### 运行KNN算法

```bash
# 运行自实现的KNN算法
python models/knn_classify_iris.py

# 运行改进版KNN算法
python models/knn_classify_iris2.0.py

# 运行sklearn的KNN实现
python models/knn\(sklearn\)_classify_iris.py
```

### 运行SVM算法

```bash
# 运行自实现的SVM算法
python models/svm_classify_iris.py

# 运行sklearn的SVM实现
python models/svm\(sklearn\)_classify_iris.py
```

### 数据可视化

```bash
# 绘制鸢尾花数据特征散点图
python visualization/plot_iris.py
```

## 算法比较

| 算法 | 实现方式 | 准确率 | 优点 | 缺点 |
|------|---------|-------|------|------|
| KNN | 自实现基础版 | ~34% | 简单直观 | 准确率低 |
| KNN | 自实现改进版 | 91%-97% | 高准确率，加权策略有效 | 类别划分区间可能不够合理 |
| KNN | sklearn | ~97% | 高准确率，实现简单 | 黑盒，不利于理解算法 |
| SVM | 自实现 | 视参数而定 | 理解算法原理 | 实现复杂，优化困难 |
| SVM | sklearn | ~97% | 高准确率，参数可调 | 黑盒，不利于理解算法 |

## 改进方向

1. **KNN算法**：
   - 在加权平均结果到最终类别的映射中，当前使用的区间划分为(0,0.5],(0.5,1.5],(1.5,2]，可以探索更合理的划分方式
   - 尝试不同的距离计算方法，如曼哈顿距离、闵可夫斯基距离等
   - 实现自动选择最优K值的方法

2. **SVM算法**：
   - 改进优化方法，如使用SMO算法
   - 尝试不同的核函数
   - 实现多类分类的不同策略，如一对一、一对多等

3. **通用改进**：
   - 实现交叉验证
   - 添加特征选择和特征工程
   - 尝试集成学习方法

## 参考资料

- [UCI Machine Learning Repository: Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris)
- [K近邻算法 - 维基百科](https://zh.wikipedia.org/wiki/K%E8%BF%91%E9%82%BB%E7%AE%97%E6%B3%95)
- [支持向量机 - 维基百科](https://zh.wikipedia.org/wiki/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA)

