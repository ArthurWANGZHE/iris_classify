"""KNN分类器实现

此模块实现了K近邻算法来对鸢尾花数据集进行分类。
"""

import sys
import os

# 添加项目根目录到系统路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import random
import heapq

# 导入工具类
from utils.data_loader import load_data, preprocess_data, split_data_by_class
from utils.metrics import euclidean_distance, accuracy, confusion_matrix

# 导入数据
data = pd.read_csv('../data/iris.data', sep=',', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
data = np.array(data)

# 按类别分割数据
setosa_data = []
versicolor_data = []
virginica_data = []
for i in range(50):
    setosa_data.append(data[i])
for i in range(50, 100):
    versicolor_data.append(data[i])
for i in range(100, 150):
    virginica_data.append(data[i])

# 将花种类赋值为数字标签
# 山鸢尾 (Iris-setosa) -> 0
# 变色鸢尾 (Iris-versicolor) -> 1
# 维吉尼亚鸢尾 (Iris-virginica) -> 2
for i in range(50):
    setosa_data[i][4] = 0
    versicolor_data[i][4] = 1
    virginica_data[i][4] = 2
# 划分训练集和测试集（取70%作为训练集，剩下30%作为测试集）
# 确保每个类别的样本在训练集和测试集中的比例相同
train_data = []
test_data = []

# 对每个类别分别进行划分
for class_data in [setosa_data, versicolor_data, virginica_data]:
    # 创建索引列表
    indices = list(range(50))
    # 随机选择70%的样本作为训练集
    train_indices = random.sample(indices, 35)
    
    # 将选中的样本添加到训练集
    for i in train_indices:
        train_data.append(class_data[i])
        indices.remove(i)
    
    # 将剩余的样本添加到测试集
    for i in indices:
        test_data.append(class_data[i])

# 打印数据集大小
print(f"训练集大小: {len(train_data)}")
print(f"测试集大小: {len(test_data)}")


def knn_classify(train_data, test_instance, k=3):
    """
    使用KNN算法对测试样本进行分类
    
    参数:
        train_data (list): 训练数据集
        test_instance (list): 测试样本
        k (int): 近邻数量
        
    返回:
        int: 预测的类别
    """
    # 计算测试样本与所有训练样本的距离
    distances = []
    for train_instance in train_data:
        # 提取特征（前4个值）
        train_features = train_instance[:4]
        test_features = test_instance[:4]
        
        # 计算欧几里得距离
        dist = euclidean_distance(train_features, test_features)
        
        # 存储距离和对应的类别
        distances.append((dist, train_instance[4]))
    
    # 对距离进行排序
    distances.sort(key=lambda x: x[0])
    
    # 获取最近的k个邻居
    nearest_neighbors = distances[:k]
    
    # 统计每个类别的数量
    class_counts = {0: 0, 1: 0, 2: 0}
    for _, label in nearest_neighbors:
        class_counts[label] += 1
    
    # 找出出现次数最多的类别
    max_count = -1
    predicted_class = -1
    for class_label, count in class_counts.items():
        if count > max_count:
            max_count = count
            predicted_class = class_label
    
    return predicted_class

# 设置K值
k = 3

# 对测试集中的每个样本进行预测
y_true = []
y_pred = []

for i, test_instance in enumerate(test_data):
    # 提取真实标签
    true_label = test_instance[4]
    y_true.append(true_label)
    
    # 使用KNN算法预测类别
    predicted_label = knn_classify(train_data, test_instance, k)
    y_pred.append(predicted_label)

# 计算准确率
acc = accuracy(y_true, y_pred)
print(f"K值: {k}")
print(f"准确率: {acc:.4f}")

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_true, y_pred)
print("\n混淆矩阵:")
print(conf_matrix)

# 如果需要可视化结果，可以使用以下代码
# from visualization.visualize import plot_confusion_matrix
# plot_confusion_matrix(conf_matrix)


