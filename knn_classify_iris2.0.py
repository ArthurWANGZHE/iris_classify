import pandas as pd
import numpy as np
import random

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

def calDistance(test, train):
    dist = ((float(test[0]) - float(train[0])) ** 2 + (float(test[1]) - float(train[1])) ** 2
            + (float(test[2]) - float(train[2])) ** 2 + (float(test[3]) - float(train[3])) ** 2
            ) ** 0.5
    return dist

labeltest = []
for i in range(45):
    distance = []
    labelTrain = []
    labeljudge = [0.0, 0.0, 0.0]
    for j in range(105):
        distance.append(calDistance(test_data[i], train_data[j]))
        labelTrain.append(train_data[j][4])
    distance2 = sorted(distance)
    ad=[]
    ar=[]
    for p in range(0, 3):
        a = distance2[p]
        b = distance.index(a)
        ad.append(a)
        ar.append(train_data[b][4])

    labelType=(ad[0]*ar[0]+ad[1]*ar[1]+ad[2]*ar[2])/(ad[0]+ad[1]+ad[2])


    if labelType <=0.5:
        labeltest.append(0)
    elif labelType <=1.5 and labelType>0.5:
        labeltest.append(1)
    elif labelType >1.5:
        labeltest.append(2)

print(labeltest)
print(answer)
correct = 0
for i in range(45):
    if labeltest[i] == answer[i]:
        correct +=1

print(correct/45)