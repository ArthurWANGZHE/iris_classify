import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('iris.data', sep=' ')
print(data.shape[0])
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

x_setosa = []
y_setosa = []
x_versicolor = []
y_versicolor = []
x_virginica=[]
y_virginica=[]
for i in range(49):
    x_setosa.append(setosa_data[i][0])
    y_setosa.append(setosa_data[i][1])
    x_versicolor.append(versicolor_data[i][0])
    y_versicolor.append(versicolor_data[i][1])
    x_virginica.append(virginica_data[i][0])
    y_virginica.append(virginica_data[i][1])

x = np.array(x_setosa)
y = np.array(y_setosa)
plt.scatter(x, y, color="blue")

x = np.array(x_versicolor)
y = np.array(y_versicolor)
plt.scatter(x, y, color="red")

x = np.array(x_virginica)
y = np.array(y_virginica)
plt.scatter(x, y, color="orange")

print(setosa_data[0])
