from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn import model_selection

iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=0.3)

KNN=KNeighborsClassifier(n_neighbors=10)
KNN.fit(x_train,y_train)
score=KNN.score(x_test,y_test)
print(score)
