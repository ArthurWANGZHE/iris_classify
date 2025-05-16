"""改进版KNN分类器实现

此模块实现了改进版的K近邻算法来对鸢尾花数据集进行分类。
与基础版本相比，改进了数据划分方式和距离计算方法。
"""

import sys
import os

# 添加项目根目录到系统路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import random

# 导入工具类
from utils.metrics import euclidean_distance, accuracy, confusion_matrix

# 加载数据
print("正在加载鸢尾花数据集...")
data = pd.read_csv('../data/iris.data', sep=',', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
data = np.array(data)

# 将花种类赋值为数字标签
# 山鸢尾 (Iris-setosa) -> 0
# 变色鸢尾 (Iris-versicolor) -> 1
# 维吉尼亚鸢尾 (Iris-virginica) -> 2
print("将类别转换为数字标签...")
for i in range(50):
    data[i][4]=0
for i in range(50, 100):
    data[i][4]=1
for i in range(100, 150):
    data[i][4]=2

# 划分训练集和测试集（取70%作为训练集，剩下30%作为测试集）
# 与基础版本不同，这里不再划分类别，而是直接从所有数据中随机选择
print("划分训练集和测试集...")
train_data = []
test_data = []

# 创建所有数据的索引列表
list0 = list(range(150))

# 随机选择70%的样本作为训练集
train0 = random.sample(list0, 105)

# 将选中的样本添加到训练集
for i in train0:
    train_data.append(data[i])
    list0.remove(i)

# 将剩余的样本添加到测试集
for i in list0:
    test_data.append(data[i])

# 提取测试集的真实标签作为参考答案
answer = []
for i in range(len(test_data)):
    answer.append(test_data[i][4])

print(f"训练集大小: {len(train_data)}")
print(f"测试集大小: {len(test_data)}")

def calDistance(test, train):
    """
    计算两个样本之间的欧几里得距离
    
    参数:
        test (list/array): 测试样本的特征
        train (list/array): 训练样本的特征
        
    返回:
        float: 两个样本之间的欧几里得距离
    """
    # 计算欧几里得距离: 各特征差的平方和的平方根
    dist = ((float(test[0]) - float(train[0])) ** 2 + 
            (float(test[1]) - float(train[1])) ** 2 +
            (float(test[2]) - float(train[2])) ** 2 + 
            (float(test[3]) - float(train[3])) ** 2
           ) ** 0.5
    return dist

def knn_classify_weighted(train_data, test_instance, k=3):
    """
    使用加权KNN算法对测试样本进行分类
    
    参数:
        train_data (list): 训练数据集
        test_instance (list): 测试样本
        k (int): 近邻数量
        
    返回:
        float: 加权平均后的类别值（需要进一步处理才能得到最终类别）
    """
    # 计算测试样本与所有训练样本的距离
    distances = []
    labels = []
    
    for train_instance in train_data:
        # 计算距离
        dist = calDistance(test_instance, train_instance)
        distances.append(dist)
        # 存储对应的类别
        labels.append(train_instance[4])
    
    # 对距离进行排序
    sorted_distances = sorted(distances)
    
    # 获取最近的k个邻居的距离和标签
    nearest_distances = []
    nearest_labels = []
    
    for p in range(k):
        # 获取排序后的第p个距离
        dist = sorted_distances[p]
        # 找到该距离在原始列表中的索引
        idx = distances.index(dist)
        # 添加到最近邻居列表
        nearest_distances.append(dist)
        nearest_labels.append(labels[idx])
    
    # 计算加权平均类别
    # 使用距离的倒数作为权重，距离越近权重越大
    # 为避免除以0的情况，可以直接使用距离作为权重
    weighted_sum = sum([dist * label for dist, label in zip(nearest_distances, nearest_labels)])
    weight_sum = sum(nearest_distances)
    
    # 返回加权平均类别
    return weighted_sum / weight_sum

# 对测试集中的每个样本进行预测
print("使用加权KNN算法进行预测...")
labeltest = []

# 设置K值
k = 3

for i in range(len(test_data)):
    # 使用加权KNN算法得到加权平均类别值
    labelType = knn_classify_weighted(train_data, test_data[i], k)

    # 根据加权平均类别值确定最终类别
    # 类别划分区间: [0, 0.5] -> 0, (0.5, 1.5] -> 1, (1.5, 2] -> 2
    if labelType <= 0.5:
        labeltest.append(0)  # 山鸢尾 (Setosa)
    elif labelType <= 1.5 and labelType > 0.5:
        labeltest.append(1)  # 变色鸢尾 (Versicolor)
    elif labelType > 1.5:
        labeltest.append(2)  # 维吉尼亚鸢尾 (Virginica)

# 打印预测结果和真实标签
print("预测结果:")
print(labeltest)
print("真实标签:")
print(answer)

# 计算准确率
correct = 0
for i in range(len(test_data)):
    if labeltest[i] == answer[i]:
        correct += 1

acc = correct / len(test_data)
print(f"K值: {k}")
print(f"准确率: {acc:.4f}")

# 计算混淆矩阵
conf_matrix = confusion_matrix(answer, labeltest)
print("\n混淆矩阵:")
print(conf_matrix)

# 如果需要可视化结果，可以使用以下代码
# from visualization.visualize import plot_confusion_matrix
# plot_confusion_matrix(conf_matrix)