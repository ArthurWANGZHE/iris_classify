import pandas as pd
import numpy as np
import random
import heapq
# 导入数据
data = pd.read_csv('iris.data', sep=' ')
data = pd.read_csv('iris.data', sep=',', names=[i for i in range(5)])
data = np.array(data)
setosa_data = []
versicolor_data = []
virginica_data = []
for i in range(50):
    setosa_data.append(data[i])
for i in range(50, 100):
    versicolor_data.append(data[i])
for i in range(100, 150):
    virginica_data.append(data[i])

# 将花种类赋值
for i in range(50):
    setosa_data[i][4] = 0
    versicolor_data[i][4] = 1
    virginica_data[i][4] = 2
# 划分训练集和测试集（取70%作为训练集，剩下30%作为测试集）
train_data =[]
test_data=[]
list0 = list(range(50))
train0 = random.sample(list0, 35)
for i in train0:
    train_data.append(setosa_data[i])
    list0.remove(i)
for i in list0:
    test_data.append(setosa_data[i])
list1 = list(range(50))
train1 = random.sample(list1, 35)
for i in train1:
    train_data.append(versicolor_data[i])
    list1.remove(i)
for i in list1:
    test_data.append(versicolor_data[i])
list2 = list(range(50))
train2 = random.sample(list2, 35)
for i in train2:
    train_data.append(virginica_data[i])
    list2.remove(i)
for i in list2:
    test_data.append(virginica_data[i])


k = 3
# 计算距离
correct = 0
for i in range(44):
    distance = []
    results = []
    r=0
    for j in range(104):
        d = abs(((test_data[i][0] - train_data[j][0]) ** 2 + (test_data[i][1] - train_data[j][1]) ** 2 +
                      (test_data[i][2] - train_data[j][2]) ** 2 + (test_data[i][3] - train_data[j][3]) ** 2))**0.5
        distance.append(d)
        min_number = heapq.nsmallest(k, distance)
        min_index = []
        for t in min_number:
            index = distance.index(t)
            min_index.append(index)
    for i in min_index:
        results.append(train_data[i][4])


    a = {0:0,1:0,2:0}
    for i in results:
        if results.count(i) > 1:
             a[i] = results.count(i)

    if a[0]>a[1] and a[0]>a[2]:
        r=0
    elif a[1]>a[0] and a[1]>a[2]:
        r=1
    else: r= 2
    if (test_data[i][4]) == r:
        correct += 1

correctness = correct / 44
print(k)
print(correctness)


