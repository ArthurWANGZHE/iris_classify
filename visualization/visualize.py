"""
可视化工具模块

此模块提供了用于可视化鸢尾花数据集和分类结果的函数。
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_iris_features(data, feature_indices=(0, 1), class_labels=None):
    """
    绘制鸢尾花数据集的特征散点图
    
    参数:
        data (numpy.ndarray): 鸢尾花数据集
        feature_indices (tuple): 要绘制的特征索引，默认为(0, 1)，即花萼长度和花萼宽度
        class_labels (list): 类别标签，默认为None，使用数字标签
    """
    if class_labels is None:
        class_labels = ["山鸢尾 (Setosa)", "变色鸢尾 (Versicolor)", "维吉尼亚鸢尾 (Virginica)"]
    
    # 分离不同类别的数据
    setosa_data = data[:50]
    versicolor_data = data[50:100]
    virginica_data = data[100:150]
    
    # 提取要绘制的特征
    feature_names = ["花萼长度 (Sepal Length)", "花萼宽度 (Sepal Width)", 
                     "花瓣长度 (Petal Length)", "花瓣宽度 (Petal Width)"]
    
    x_feature = feature_indices[0]
    y_feature = feature_indices[1]
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    
    # 山鸢尾数据点
    x_setosa = [setosa_data[i][x_feature] for i in range(len(setosa_data))]
    y_setosa = [setosa_data[i][y_feature] for i in range(len(setosa_data))]
    plt.scatter(x_setosa, y_setosa, color="blue", label=class_labels[0])
    
    # 变色鸢尾数据点
    x_versicolor = [versicolor_data[i][x_feature] for i in range(len(versicolor_data))]
    y_versicolor = [versicolor_data[i][y_feature] for i in range(len(versicolor_data))]
    plt.scatter(x_versicolor, y_versicolor, color="red", label=class_labels[1])
    
    # 维吉尼亚鸢尾数据点
    x_virginica = [virginica_data[i][x_feature] for i in range(len(virginica_data))]
    y_virginica = [virginica_data[i][y_feature] for i in range(len(virginica_data))]
    plt.scatter(x_virginica, y_virginica, color="green", label=class_labels[2])
    
    # 添加标题和标签
    plt.title("鸢尾花数据集特征可视化")
    plt.xlabel(feature_names[x_feature])
    plt.ylabel(feature_names[y_feature])
    plt.legend()
    plt.grid(True)
    
    plt.show()

def plot_all_features(data):
    """
    绘制鸢尾花数据集的所有特征组合散点图
    
    参数:
        data (numpy.ndarray): 鸢尾花数据集
    """
    feature_combinations = [
        (0, 1),  # 花萼长度 vs 花萼宽度
        (2, 3),  # 花瓣长度 vs 花瓣宽度
        (0, 2),  # 花萼长度 vs 花瓣长度
        (1, 3),  # 花萼宽度 vs 花瓣宽度
        (0, 3),  # 花萼长度 vs 花瓣宽度
        (1, 2)   # 花萼宽度 vs 花瓣长度
    ]
    
    for feature_pair in feature_combinations:
        plot_iris_features(data, feature_pair)

def plot_confusion_matrix(confusion_mat, class_names=None):
    """
    绘制混淆矩阵
    
    参数:
        confusion_mat (numpy.ndarray): 混淆矩阵
        class_names (list): 类别名称，默认为None
    """
    if class_names is None:
        class_names = ["山鸢尾", "变色鸢尾", "维吉尼亚鸢尾"]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("混淆矩阵")
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 在每个单元格中添加数字
    thresh = confusion_mat.max() / 2.
    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            plt.text(j, i, format(confusion_mat[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if confusion_mat[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()

def plot_decision_boundary(X, y, model, feature_indices=(0, 1), mesh_step_size=0.02):
    """
    绘制决策边界
    
    参数:
        X (numpy.ndarray): 特征数据
        y (numpy.ndarray): 标签数据
        model: 分类模型，必须有predict方法
        feature_indices (tuple): 要绘制的特征索引，默认为(0, 1)
        mesh_step_size (float): 网格步长，默认为0.02
    """
    # 提取要绘制的特征
    X = np.array(X)
    feature_names = ["花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度"]
    
    x_index = feature_indices[0]
    y_index = feature_indices[1]
    
    # 提取选定的特征
    X_selected = X[:, [x_index, y_index]]
    
    # 设置图形大小
    plt.figure(figsize=(10, 8))
    
    # 定义网格范围
    x_min, x_max = X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1
    y_min, y_max = X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1
    
    # 创建网格点
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))
    
    # 为网格点创建特征向量
    Z_points = np.c_[xx.ravel(), yy.ravel()]
    
    # 如果模型需要所有特征，则需要填充其他特征
    if hasattr(model, 'predict'):
        if X.shape[1] > 2:
            # 创建一个平均特征向量
            avg_features = np.mean(X, axis=0)
            full_features = np.zeros((Z_points.shape[0], X.shape[1]))
            
            # 填充所有特征
            for i in range(X.shape[1]):
                if i == x_index:
                    full_features[:, i] = Z_points[:, 0]
                elif i == y_index:
                    full_features[:, i] = Z_points[:, 1]
                else:
                    full_features[:, i] = avg_features[i]
            
            # 预测网格点的类别
            Z = model.predict(full_features)
        else:
            # 如果模型只需要两个特征
            Z = model.predict(Z_points)
    else:
        # 如果模型没有predict方法，尝试直接调用
        if X.shape[1] > 2:
            # 创建一个平均特征向量
            avg_features = np.mean(X, axis=0)
            full_features = np.zeros((Z_points.shape[0], X.shape[1]))
            
            # 填充所有特征
            for i in range(X.shape[1]):
                if i == x_index:
                    full_features[:, i] = Z_points[:, 0]
                elif i == y_index:
                    full_features[:, i] = Z_points[:, 1]
                else:
                    full_features[:, i] = avg_features[i]
            
            # 预测网格点的类别
            Z = model(full_features)
        else:
            # 如果模型只需要两个特征
            Z = model(Z_points)
    
    # 将预测结果重塑为网格形状
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.3)
    
    # 绘制训练点
    colors = ['blue', 'red', 'green']
    class_names = ["山鸢尾", "变色鸢尾", "维吉尼亚鸢尾"]
    
    for i, color in enumerate(colors):
        idx = np.where(y == i)
        plt.scatter(X_selected[idx, 0], X_selected[idx, 1], c=color, label=class_names[i],
                   edgecolor='black', s=50)
    
    plt.title("决策边界")
    plt.xlabel(feature_names[x_index])
    plt.ylabel(feature_names[y_index])
    plt.legend()
    plt.grid(True)
    
    plt.show()
