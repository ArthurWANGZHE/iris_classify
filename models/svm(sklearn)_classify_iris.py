"""使用sklearn实现SVM分类器

此模块使用sklearn库的SVM实现了支持向量机算法来对鸢尾花数据集进行分类。
"""

import sys
import os

# 添加项目根目录到系统路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入sklearn相关模块
from sklearn import datasets
from sklearn import model_selection
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix

# 导入可视化工具（如果需要）
# from visualization.visualize import plot_confusion_matrix

def main():
    """主函数，执行SVM分类"""
    # 加载鸢尾花数据集
    print("正在加载鸢尾花数据集...")
    iris = datasets.load_iris()
    x = iris.data  # 特征数据
    y = iris.target  # 标签数据
    
    # 划分训练集和测试集（70%训练，30%测试）
    print("划分训练集和测试集...")
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, random_state=1, test_size=0.3
    )
    
    # 创建SVM分类器
    print("创建SVM分类器...")
    # kernel: 核函数类型，'rbf'为高斯核函数
    # gamma: 核函数的系数，值越小，支持向量越少，泛化能力越强
    # C: 惩罚参数，值越大，对误分类的惩罚越重，泛化能力越弱
    # decision_function_shape: 决策函数类型，'ovo'为一对一策略
    classifier = svm.SVC(kernel='rbf', gamma=0.1, decision_function_shape='ovo', C=1)
    
    # 训练模型
    print("训练模型...")
    classifier.fit(x_train, y_train.ravel())  # ravel()将多维数组转为一维
    
    # 计算训练集和测试集的准确率
    train_score = classifier.score(x_train, y_train)
    test_score = classifier.score(x_test, y_test)
    
    print(f'SVM-训练集准确率: {train_score:.4f}')
    print(f'SVM-测试集准确率: {test_score:.4f}')
    
    # 对测试集进行预测
    y_pred = classifier.predict(x_test)
    
    # 打印分类报告
    print("\n分类报告:")
    target_names = ['山鸢尾 (Setosa)', '变色鸢尾 (Versicolor)', '维吉尼亚鸢尾 (Virginica)']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 计算混淆矩阵
    conf_mat = confusion_matrix(y_test, y_pred)
    print("\n混淆矩阵:")
    print(conf_mat)
    
    # 如果需要可视化结果，可以取消下面的注释
    # plot_confusion_matrix(conf_mat, class_names=target_names)

# 如果直接运行此脚本，则执行main函数
if __name__ == "__main__":
    main()

















