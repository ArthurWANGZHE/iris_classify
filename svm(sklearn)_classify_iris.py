from sklearn import datasets
from sklearn import model_selection
from sklearn import svm

iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=0.3)
classifier = svm.SVC(kernel='rbf', gamma=0.1, decision_function_shape='ovo', C=1)
classifier.fit(x_train, y_train.ravel())

print('SVM-输出训练集的准确率为： %.2f' % classifier.score(x_train, y_train))
print('SVM-输出测试集的准确率为:  %.2f' % classifier.score(x_test, y_test))
print('\npredict:\n', classifier.predict(x_train))

















