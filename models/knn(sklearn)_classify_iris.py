"""使用sklearn实现KNN分类器

此模块使用sklearn库的KNeighborsClassifier实现了K近邻算法来对鸢尾花数据集进行分类。
"""

import sys
import os

# 添加项目根目录到系统路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入sklearn相关模块
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix

# 导入可视化工具（如果需要）
# from visualization.visualize import plot_confusion_matrix

def main():
    """主函数，执行KNN分类"""
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
    
    # 创建KNN分类器
    print("创建KNN分类器...")
    k = 10  # 近邻数量
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # 训练模型
    print("训练模型...")
    knn.fit(x_train, y_train)
    
    # 在测试集上评估模型
    print("评估模型...")
    score = knn.score(x_test, y_test)
    print(f"KNN分类器 (k={k}) 的准确率: {score:.4f}")
    
    # 预测测试集
    y_pred = knn.predict(x_test)
    
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
