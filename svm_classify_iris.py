import numpy as np
import random
import pandas as pd
# 定义超平面函数
def hyperplane(x, w, b, v):
    return np.dot(w, x) + b

# 根据超平面函数计算分类结果
def predict(X, w, b, v):
    X = np.array(X)
    w = np.array(w)
    b = np.array(b)
    v = np.array(v)
    hyperplane_value = np.apply_along_axis(hyperplane, 1, X, w, b, v)
    return np.sign(hyperplane_value)


#根据分类结果计算误差

def loss(X, Y, w, b, v):
    X = np.array(X)
    Y = np.array(Y)
    y_pred = predict(X, w, b, v)
    errors = np.where(y_pred != Y, 1, 0)
    return np.sum(errors)


#计算梯度

def gradient(X, Y, w, b,v):
    X = np.array(X)
    Y = np.array(Y)
    y_pred = predict(X, w, b, v)
    dw = np.zeros(w.shape)
    db = 0
    dv = 0
    for i in range(X.shape[0]):
       if y_pred[i] * (np.dot(w, X[i]) + b + np.dot(v, X[i])) < 1:
           dw += -Y[i] * X[i]
           db += -Y[i]
           dv += -Y[i] * X[i]
           return dw, db, dv

def update(X, Y, w, b, v,learning_rate):
    dw, db, dv = gradient(X, Y, w, b, v)
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
        X = np.array(X)
        Y = np.array(Y)
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.v = np.zeros(X.shape[1])
        for _ in range(1000):
            self.w, self.b, self.v = update(X, Y, self.w, self.b, self.v, self.learning_rate)
            loss_value = loss(X, Y, self.w, self.b, self.v)


    def predict(self, X):
        X = np.array(X)
        return predict(X, self.w, self.b, self.v)

data = pd.read_csv('iris.data', sep=',', names=[i for i in range(5)])
data = np.array(data)

for i in range(50):
    data[i][4]=0
for i in range(50, 100):
    data[i][4]=1
for i in range(100, 150):
    data[i][4]=2

train_data =[]
test_data=[]
list0 = list(range(150))
train0 = random.sample(list0, 105)
for i in train0:
    train_data.append(data[i])
    list0.remove(i)
for i in list0:
    test_data.append(data[i])
answer = []
for i in range(45):
    answer.append(test_data[i][4])

testdata=[]
for i in range(45):
    test_data[i]=list(test_data[i])
    test_data[i].pop()
    testdata.append(test_data[i])


X=[]
Y=[]
for i in range(105):
    Y.append(train_data[i][4])
for i in range(105):
    train_data[i]=list(train_data[i])
    train_data[i].pop()
    X.append(train_data[i])


svm = SVM()
svm.fit(X, Y)

print("w: ", svm.w)
print("b: ", svm.b)
print("v: ", svm.v)


X_test = testdata
Y_test = svm.predict(X_test)
print("Prediction: ", Y_test)
print(answer)
