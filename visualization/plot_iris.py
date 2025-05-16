"""鸢尾花数据集可视化模块

此模块提供了用于可视化鸢尾花数据集的函数。
"""

import sys
import os

# 添加项目根目录到系统路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
def load_iris_data():
    """
    加载鸢尾花数据集并按类别分组
    
    返回:
        tuple: 三种鸢尾花的数据 (setosa_data, versicolor_data, virginica_data)
    """
    # 加载数据
    data = pd.read_csv('../data/iris.data', sep=',', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
    data_array = np.array(data)
    
    # 按类别分组数据
    setosa_data = []      # 山鸢尾
    versicolor_data = []  # 变色鸢尾
    virginica_data = []   # 维吉尼亚鸢尾
    
    # 山鸢尾数据（前50行）
    for i in range(50):
        setosa_data.append(data_array[i])
    # 变色鸢尾数据（中间50行）
    for i in range(50, 100):
        versicolor_data.append(data_array[i])
    # 维吉尼亚鸢尾数据（后50行）
    for i in range(100, 150):
        virginica_data.append(data_array[i])
        
    return setosa_data, versicolor_data, virginica_data

# 加载数据
setosa_data, versicolor_data, virginica_data = load_iris_data()

def plot_sepal_features():
    """
    绘制花萌长度和花萌宽度的散点图
    """
    # 提取花萌长度和花萌宽度特征
    x_setosa = []      # 山鸢尾的花萌长度
    y_setosa = []      # 山鸢尾的花萌宽度
    x_versicolor = []  # 变色鸢尾的花萌长度
    y_versicolor = []  # 变色鸢尾的花萌宽度
    x_virginica = []   # 维吉尼亚鸢尾的花萌长度
    y_virginica = []   # 维吉尼亚鸢尾的花萌宽度
    
    # 提取每个类别的特征
    for i in range(len(setosa_data)):
        x_setosa.append(setosa_data[i][0])      # 花萌长度
        y_setosa.append(setosa_data[i][1])      # 花萌宽度
        x_versicolor.append(versicolor_data[i][0])  # 花萌长度
        y_versicolor.append(versicolor_data[i][1])  # 花萌宽度
        x_virginica.append(virginica_data[i][0])   # 花萌长度
        y_virginica.append(virginica_data[i][1])   # 花萌宽度
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制山鸢尾数据点
    x = np.array(x_setosa)
    y = np.array(y_setosa)
    plt.scatter(x, y, color="blue", label="山鸢尾 (Setosa)")
    
    # 绘制变色鸢尾数据点
    x = np.array(x_versicolor)
    y = np.array(y_versicolor)
    plt.scatter(x, y, color="red", label="变色鸢尾 (Versicolor)")
    
    # 绘制维吉尼亚鸢尾数据点
    x = np.array(x_virginica)
    y = np.array(y_virginica)
    plt.scatter(x, y, color="green", label="维吉尼亚鸢尾 (Virginica)")
    
    # 添加标题和标签
    plt.title("鸢尾花数据集 - 花萌特征可视化")
    plt.xlabel("花萌长度 (Sepal Length)")
    plt.ylabel("花萌宽度 (Sepal Width)")
    plt.legend()
    plt.grid(True)
    
    # 显示图形
    plt.show()

def plot_petal_features():
    """
    绘制花瓣长度和花瓣宽度的散点图
    """
    # 提取花瓣长度和花瓣宽度特征
    x_setosa = []      # 山鸢尾的花瓣长度
    y_setosa = []      # 山鸢尾的花瓣宽度
    x_versicolor = []  # 变色鸢尾的花瓣长度
    y_versicolor = []  # 变色鸢尾的花瓣宽度
    x_virginica = []   # 维吉尼亚鸢尾的花瓣长度
    y_virginica = []   # 维吉尼亚鸢尾的花瓣宽度
    
    # 提取每个类别的特征
    for i in range(len(setosa_data)):
        x_setosa.append(setosa_data[i][2])      # 花瓣长度
        y_setosa.append(setosa_data[i][3])      # 花瓣宽度
        x_versicolor.append(versicolor_data[i][2])  # 花瓣长度
        y_versicolor.append(versicolor_data[i][3])  # 花瓣宽度
        x_virginica.append(virginica_data[i][2])   # 花瓣长度
        y_virginica.append(virginica_data[i][3])   # 花瓣宽度
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制山鸢尾数据点
    x = np.array(x_setosa)
    y = np.array(y_setosa)
    plt.scatter(x, y, color="blue", label="山鸢尾 (Setosa)")
    
    # 绘制变色鸢尾数据点
    x = np.array(x_versicolor)
    y = np.array(y_versicolor)
    plt.scatter(x, y, color="red", label="变色鸢尾 (Versicolor)")
    
    # 绘制维吉尼亚鸢尾数据点
    x = np.array(x_virginica)
    y = np.array(y_virginica)
    plt.scatter(x, y, color="green", label="维吉尼亚鸢尾 (Virginica)")
    
    # 添加标题和标签
    plt.title("鸢尾花数据集 - 花瓣特征可视化")
    plt.xlabel("花瓣长度 (Petal Length)")
    plt.ylabel("花瓣宽度 (Petal Width)")
    plt.legend()
    plt.grid(True)
    
    # 显示图形
    plt.show()

# 如果直接运行此脚本，则绘制花萌特征图和花瓣特征图
if __name__ == "__main__":
    print("绘制花萌特征图...")
    plot_sepal_features()
    
    print("绘制花瓣特征图...")
    plot_petal_features()
