"""
数据加载和预处理工具模块

此模块提供了用于加载和预处理鸢尾花数据集的函数。
"""
import pandas as pd
import numpy as np
import random

def load_data(file_path='../data/iris.data'):
    """
    加载鸢尾花数据集
    
    参数:
        file_path (str): 数据文件路径
        
    返回:
        numpy.ndarray: 加载的数据集
    """
    data = pd.read_csv(file_path, sep=',', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
    return data

def preprocess_data(data):
    """
    预处理数据，将花的类别转换为数字标签
    
    参数:
        data (pandas.DataFrame): 原始数据
        
    返回:
        numpy.ndarray: 预处理后的数据
    """
    # 将数据转换为numpy数组
    data_array = np.array(data)
    
    # 将花的类别转换为数字标签
    # Iris-setosa -> 0
    # Iris-versicolor -> 1
    # Iris-virginica -> 2
    for i in range(50):
        data_array[i][4] = 0
    for i in range(50, 100):
        data_array[i][4] = 1
    for i in range(100, 150):
        data_array[i][4] = 2
        
    return data_array

def split_data_by_class(data_array):
    """
    按类别分割数据
    
    参数:
        data_array (numpy.ndarray): 预处理后的数据
        
    返回:
        tuple: 三种花的数据 (setosa_data, versicolor_data, virginica_data)
    """
    setosa_data = []
    versicolor_data = []
    virginica_data = []
    
    for i in range(50):
        setosa_data.append(data_array[i])
    for i in range(50, 100):
        versicolor_data.append(data_array[i])
    for i in range(100, 150):
        virginica_data.append(data_array[i])
        
    return setosa_data, versicolor_data, virginica_data

def split_train_test(data_array, train_ratio=0.7):
    """
    将数据分割为训练集和测试集
    
    参数:
        data_array (numpy.ndarray): 预处理后的数据
        train_ratio (float): 训练集比例
        
    返回:
        tuple: 训练集和测试集 (train_data, test_data)
    """
    train_data = []
    test_data = []
    
    # 随机分割数据
    list_all = list(range(150))
    train_indices = random.sample(list_all, int(150 * train_ratio))
    
    for i in train_indices:
        train_data.append(data_array[i])
        list_all.remove(i)
    
    for i in list_all:
        test_data.append(data_array[i])
    
    return train_data, test_data

def split_train_test_balanced(setosa_data, versicolor_data, virginica_data, train_ratio=0.7):
    """
    将数据按类别平衡地分割为训练集和测试集
    
    参数:
        setosa_data (list): 山鸢尾数据
        versicolor_data (list): 变色鸢尾数据
        virginica_data (list): 维吉尼亚鸢尾数据
        train_ratio (float): 训练集比例
        
    返回:
        tuple: 训练集和测试集 (train_data, test_data)
    """
    train_data = []
    test_data = []
    
    # 计算每个类别的训练样本数量
    train_samples_per_class = int(50 * train_ratio)
    
    # 对每个类别进行分割
    for class_data in [setosa_data, versicolor_data, virginica_data]:
        indices = list(range(50))
        train_indices = random.sample(indices, train_samples_per_class)
        
        for i in train_indices:
            train_data.append(class_data[i])
            indices.remove(i)
        
        for i in indices:
            test_data.append(class_data[i])
    
    return train_data, test_data
