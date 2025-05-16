"""支持向量机 (SVM) 分类器实现

此模块实现了支持向量机算法来对鸢尾花数据集进行分类。
"""

import sys
import os

# 添加项目根目录到系统路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import pandas as pd

# 导入工具类
from utils.metrics import accuracy, confusion_matrix

# 定义超平面函数
def hyperplane(x, w, b, v):
    """
    计算超平面函数值
    
    参数:
        x (numpy.ndarray): 输入样本的特征向量
        w (numpy.ndarray): 权重向量
        b (float): 偏置项
        v (numpy.ndarray): 附加权重向量（用于非线性映射）
        
    返回:
        float: 超平面函数值
    """
    return np.dot(w, x) + b

# 根据超平面函数计算分类结果
def predict(X, w, b, v):
    """
    预测样本的类别
    
    参数:
        X (numpy.ndarray): 输入样本的特征矩阵
        w (numpy.ndarray): 权重向量
        b (float): 偏置项
        v (numpy.ndarray): 附加权重向量
        
    返回:
        numpy.ndarray: 预测的类别标签（+1或-1）
    """
    X = np.array(X)
    w = np.array(w)
    b = np.array(b)
    v = np.array(v)
    # 对每个样本计算超平面函数值
    hyperplane_value = np.apply_along_axis(hyperplane, 1, X, w, b, v)
    # 返回符号作为预测类别
    return np.sign(hyperplane_value)


#根据分类结果计算误差
def loss(X, Y, w, b, v):
    """
    计算分类误差
    
    参数:
        X (numpy.ndarray): 输入样本的特征矩阵
        Y (numpy.ndarray): 真实类别标签
        w (numpy.ndarray): 权重向量
        b (float): 偏置项
        v (numpy.ndarray): 附加权重向量
        
    返回:
        float: 误分类样本数量
    """
    X = np.array(X)
    Y = np.array(Y)
    # 预测类别
    y_pred = predict(X, w, b, v)
    # 计算误分类样本数量
    errors = np.where(y_pred != Y, 1, 0)
    return np.sum(errors)


# 计算梯度
def gradient(X, Y, w, b, v):
    """
    计算SVM目标函数对参数的梯度
    
    参数:
        X (numpy.ndarray): 输入样本的特征矩阵
        Y (numpy.ndarray): 真实类别标签
        w (numpy.ndarray): 权重向量
        b (float): 偏置项
        v (numpy.ndarray): 附加权重向量
        
    返回:
        tuple: 各参数的梯度 (dw, db, dv)
    """
    X = np.array(X)
    Y = np.array(Y)
    # 预测类别
    y_pred = predict(X, w, b, v)
    # 初始化梯度
    dw = np.zeros(w.shape)
    db = 0
    dv = 0
    
    # 计算梯度
    for i in range(X.shape[0]):
        # 对于边界上或分类错误的样本计算梯度
        if y_pred[i] * (np.dot(w, X[i]) + b + np.dot(v, X[i])) < 1:
            dw += -Y[i] * X[i]
            db += -Y[i]
            dv += -Y[i] * X[i]
    
    return dw, db, dv

def update(X, Y, w, b, v, learning_rate):
    """
    使用梯度下降更新参数
    
    参数:
        X (numpy.ndarray): 输入样本的特征矩阵
        Y (numpy.ndarray): 真实类别标签
        w (numpy.ndarray): 当前权重向量
        b (float): 当前偏置项
        v (numpy.ndarray): 当前附加权重向量
        learning_rate (float): 学习率
        
    返回:
        tuple: 更新后的参数 (w, b, v)
    """
    # 计算梯度
    dw, db, dv = gradient(X, Y, w, b, v)
    # 使用梯度下降更新参数
    w = w - learning_rate * dw
    b = b - learning_rate * db
    v = v - learning_rate * dv
    return w, b, v


class SVM:
    """
    支持向量机分类器类
    """
    def __init__(self, learning_rate=0.001, max_iterations=1000):
        """
        初始化SVM分类器
        
        参数:
            learning_rate (float): 学习率，默认为0.001
            max_iterations (int): 最大迭代次数，默认为1000
        """
        self.w = None  # 权重向量
        self.b = None  # 偏置项
        self.v = None  # 附加权重向量（用于非线性映射）
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def fit(self, X, Y):
        """
        训练SVM模型
        
        参数:
            X (numpy.ndarray): 训练样本的特征矩阵
            Y (numpy.ndarray): 训练样本的类别标签
        """
        print("开始训练SVM模型...")
        X = np.array(X)
        Y = np.array(Y)
        
        # 初始化参数
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.v = np.zeros(X.shape[1])
        
        # 迭代训练
        for iteration in range(self.max_iterations):
            # 更新参数
            self.w, self.b, self.v = update(X, Y, self.w, self.b, self.v, self.learning_rate)
            
            # 计算当前误差
            loss_value = loss(X, Y, self.w, self.b, self.v)
            
            # 每100次迭代打印一次误差
            if (iteration + 1) % 100 == 0:
                print(f"迭代 {iteration + 1}/{self.max_iterations}, 误差: {loss_value}")
                
            # 如果误差为0，提前结束训练
            if loss_value == 0:
                print(f"在迭代 {iteration + 1} 次后误差为0，提前结束训练")
                break
        
        print("模型训练完成")

    def predict(self, X):
        """
        使用训练好的模型预测样本的类别
        
        参数:
            X (numpy.ndarray): 测试样本的特征矩阵
            
        返回:
            numpy.ndarray: 预测的类别标签
        """
        X = np.array(X)
        return predict(X, self.w, self.b, self.v)

def main():
    """
    主函数，加载数据并训练SVM模型
    """
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
        data[i][4] = 0
    for i in range(50, 100):
        data[i][4] = 1
    for i in range(100, 150):
        data[i][4] = 2
    
    # 划分训练集和测试集（取70%作为训练集，剩下30%作为测试集）
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
    
    # 处理测试数据，移除标签
    testdata = []
    for i in range(len(test_data)):
        test_instance = list(test_data[i])
        test_instance.pop()  # 移除标签
        testdata.append(test_instance)
    
    # 处理训练数据，分离特征和标签
    X = []
    Y = []
    for i in range(len(train_data)):
        Y.append(train_data[i][4])  # 标签
        train_instance = list(train_data[i])
        train_instance.pop()  # 移除标签
        X.append(train_instance)  # 特征
    
    # 创建SVM模型并训练
    svm = SVM(learning_rate=0.001, max_iterations=1000)
    svm.fit(X, Y)
    
    # 打印模型参数
    print("\n模型参数:")
    print("w: ", svm.w)
    print("b: ", svm.b)
    print("v: ", svm.v)
    
    # 对测试集进行预测
    print("\n对测试集进行预测...")
    X_test = testdata
    Y_pred = svm.predict(X_test)
    
    # 打印预测结果和真实标签
    print("预测结果: ", Y_pred)
    print("真实标签: ", answer)
    
    # 计算准确率
    correct = 0
    for i in range(len(Y_pred)):
        if Y_pred[i] == answer[i]:
            correct += 1
    accuracy = correct / len(Y_pred)
    print(f"\n准确率: {accuracy:.4f}")
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(answer, Y_pred)
    print("\n混淆矩阵:")
    print(conf_matrix)
    
    # 如果需要可视化结果，可以使用以下代码
    # from visualization.visualize import plot_confusion_matrix
    # plot_confusion_matrix(conf_matrix)

# 如果直接运行此脚本，则执行main函数
if __name__ == "__main__":
    main()
