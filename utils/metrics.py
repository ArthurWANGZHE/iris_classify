"""
度量计算工具模块

此模块提供了用于计算距离和评估模型性能的函数。
"""
import numpy as np

def euclidean_distance(point1, point2):
    """
    计算两点之间的欧几里得距离
    
    参数:
        point1 (list/array): 第一个点的坐标
        point2 (list/array): 第二个点的坐标
        
    返回:
        float: 两点之间的欧几里得距离
    """
    return np.sqrt(np.sum([(float(a) - float(b)) ** 2 for a, b in zip(point1, point2)]))

def manhattan_distance(point1, point2):
    """
    计算两点之间的曼哈顿距离
    
    参数:
        point1 (list/array): 第一个点的坐标
        point2 (list/array): 第二个点的坐标
        
    返回:
        float: 两点之间的曼哈顿距离
    """
    return np.sum([abs(float(a) - float(b)) for a, b in zip(point1, point2)])

def accuracy(y_true, y_pred):
    """
    计算分类准确率
    
    参数:
        y_true (list): 真实标签
        y_pred (list): 预测标签
        
    返回:
        float: 分类准确率
    """
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

def confusion_matrix(y_true, y_pred, num_classes=3):
    """
    计算混淆矩阵
    
    参数:
        y_true (list): 真实标签
        y_pred (list): 预测标签
        num_classes (int): 类别数量
        
    返回:
        numpy.ndarray: 混淆矩阵
    """
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[int(true)][int(pred)] += 1
    return matrix

def precision_recall_f1(confusion_mat):
    """
    计算精确率、召回率和F1分数
    
    参数:
        confusion_mat (numpy.ndarray): 混淆矩阵
        
    返回:
        tuple: 精确率、召回率和F1分数的列表
    """
    num_classes = confusion_mat.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    
    for i in range(num_classes):
        # 精确率: TP / (TP + FP)
        precision[i] = confusion_mat[i, i] / max(np.sum(confusion_mat[:, i]), 1)
        
        # 召回率: TP / (TP + FN)
        recall[i] = confusion_mat[i, i] / max(np.sum(confusion_mat[i, :]), 1)
        
        # F1分数: 2 * (precision * recall) / (precision + recall)
        f1[i] = 2 * precision[i] * recall[i] / max(precision[i] + recall[i], 1e-10)
    
    return precision, recall, f1
