import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import accuracy_score

def get_iris():
    iris_data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.4, random_state=0)
    return X_train, X_test, y_train, y_test

def knn_classify(self_point, dataset, labels, k):
    distance = [np.sqrt(sum((self_point - d)**2)) for d in dataset]
    train_data = zip(distance, labels)
    train_data = sorted(train_data, key=lambda x: x[0])[:k]
    self_label = {}
    for i in train_data:
        i = str(i[1])
        self_label[i] = self_label.setdefault(i, 0) + 1
    self_label = sorted(self_label, key=self_label.get, reverse=True)
    return self_label[0]


X_train, X_test, y_train, y_test = get_iris()
size = len(y_test)
count = 0
for t in range(len(X_test)):
    y_pre = knn_classify(X_test[t], X_train, y_train, 5)
    if y_pre == str(y_test[t]):
        count += 1
# print('custom的准确率： ', count / size)

# 使用sklearn内置的KNN
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pre = knn.predict(X_test)
# print('sklearn的准确率: ', accuracy_score(y_test, pre))

from sklearn.metrics import precision_score, recall_score, f1_score
accu = precision_score(y_test, pre, average='micro')
print('sklearn的准确率: ', accu)