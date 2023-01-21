import numpy as np


# 定义超平面函数
def hyperplane(x, w, b, v):
    return np.dot(w, x) + b


# 根据超平面函数计算分类结果
def predict(X, w, b, v):  # 将输入的X转换为Numpy Array
    X = np.array(X)  # 将输入的w,b,v转换为Numpy Array
    w = np.array(w)
    b = np.array(b)
    v = np.array(v)  # 对每个样本进行分类
    hyperplane_value = np.apply_along_axis(hyperplane, 1, X, w, b, v)
    return np.sign(hyperplane_value)


#根据分类结果计算误差


def loss(X, Y, w, b, v):  # 将输入的X,Y转换为Numpy Array
    X = np.array(X)
    Y = np.array(Y)  # 计算预测结果
    y_pred = predict(X, w, b, v)  # 计算误差
    errors = np.where(y_pred != Y, 1, 0)
    return np.sum(errors)


#计算梯度


def gradient(X, Y, w, b,v):  # 将输入的X,Y转换为Numpy Array
    X = np.array(X)
    Y = np.array(Y) # 计算预测结果
    y_pred = predict(X, w, b, v) # 计算梯度
    dw = np.zeros(w.shape)
    db = 0
    dv = 0
    for i in range(X.shape[0]):
       if y_pred[i] * (np.dot(w, X[i]) + b + np.dot(v, X[i])) < 1:
           dw += -Y[i] * X[i]
           db += -Y[i]
           dv += -Y[i] * X[i]
           return dw, db, dv





def update(X, Y, w, b, v,learning_rate):  # 计算梯度
    dw, db, dv = gradient(X, Y, w, b, v) # 更新参数
    w = w - learning_rate * dw
    b = b - learning_rate * db
    v = v - learning_rate * dv
    return w, b, v





class SVM:
    def __init__(self, learning_rate=0.001):
        self.w = None
        self.b = None
        self.v = None
        self.learning_rate = learning_rate


    def fit(self, X, Y):
    # 将输入的X,Y转换为Numpy Array
        X = np.array(X)
        Y = np.array(Y)
    # 初始化参数
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.v = np.zeros(X.shape[1])
    # 迭代更新参数
        for _ in range(1000):
            self.w, self.b, self.v = update(X, Y, self.w, self.b, self.v, self.learning_rate)
        # 计算当前损失
            loss_value = loss(X, Y, self.w, self.b, self.v)


    def predict(self, X):
        # 将输入的X转换为Numpy Array
        X = np.array(X)
        # 根据超平面函数计算分类结果
        return predict(X, self.w, self.b, self.v)





X = [[2, 3], [3, 3], [2, 2], [1, 2]]
Y = [1, 1, -1, -1]


svm = SVM()
svm.fit(X, Y)

print("w: ", svm.w)
print("b: ", svm.b)
print("v: ", svm.v)


X_test = [[2, 4], [3, 4], [2, 1], [1, 1]]
Y_test = svm.predict(X_test)
print("Prediction: ", Y_test)
