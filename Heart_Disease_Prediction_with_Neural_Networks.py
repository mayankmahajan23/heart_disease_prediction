import sys
import pandas as pd
import numpy as np
import sklearn

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split

col_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'class']
cleveland_data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", names = col_names)
hungarian_data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data", names = col_names)
switzerland_data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data", names = col_names)

data = cleveland_data
data = data.append(switzerland_data)
data = data.append(hungarian_data)
print(data)

data.drop(['fbs', 'slope', 'ca', 'thal'], axis = 1, inplace = True)
data.replace('?', np.NaN, inplace = True)
data.dropna(axis = 0, inplace = True)

data = data.apply(pd.to_numeric)
for i in range(data.shape[1]):
    if (i == 8):
        data[data.columns[i]] = data[data.columns[i]].astype('float')
    else:
        data[data.columns[i]] = data[data.columns[i]].astype('int')

data.reset_index(drop = True, inplace = True)
print(data)
print(data.dtypes)

scatter_matrix(data, figsize = (20, 20))
#plt.show()
plt.clf()

data = np.array(data)

from sklearn.neighbors import KNeighborsClassifier
def knn(data, start, stop, step):
    features = data[:, :-1]
    target = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.25)

    score = []
    for i in range(start, stop, step):
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(X_train, y_train)
        score.append((knn.score(X_test, y_test), i))

    score = np.array(score, dtype = [('score', 'float64'), ('neighbours', 'int')])
    score = np.sort(score, order = 'score')
    print(score)
    print()

from sklearn.neural_network import MLPClassifier
def ann_classification(data, layer_size):
    features = data[:, :-1]
    target = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split (features, target, test_size = 0.25)

    clf = MLPClassifier(activation = 'logistic', solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = layer_size, random_state = 1, max_iter = 1e+7)

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score)

    return score
def call_ann_classification(data, start, stop, step):
    score = []
    for i in range(start, stop, step):
        score.append((ann_classification(data, (i,i,i)), i))
    score = np.array(score, dtype = [('score', 'float64'), ('nodes per layer', 'int')])
    score = np.sort(score, order = 'score')
    print(score)
    print()

from sklearn import svm
def svm_model(data, kernel, degree):
    print('in svm_model')

    features = data[:, :-1]
    target = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.25)

    clf = svm.SVC(kernel = kernel, degree = degree, C = 1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score
def linear_svc_model(data):
    features = data[:, :-1]
    target = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.25)

    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score
def call_svm(data):
    score = []

    #score.append((linear_svc_model(data), 1))
    score.append((svm_model(data, 'linear', 0), 1))
    for i in range(2, 25, 2):
        score.append((svm_model(data, 'poly', i), i))
    score.append((svm_model(data, 'rbf', 0), 9))
    score.append((svm_model(data, 'sigmoid', 0), 10))

    score = np.array(score, dtype = [('score', 'float64'), ('type', 'int')])
    score = np.sort(score, order = 'score')
    print(score)
    print()

from sklearn.tree import DecisionTreeClassifier
def decision_tree(data):
    features = data[:,:-1]
    target =  data[:,-1:]
    X_train, X_test, y_train, y_test = train_test_split (features, target, random_state = 0, test_size = 0.25)

    initial_max_depth = 10
    final_max_depth = 500
    depth_increment = 20

    score_history = []
    depth = initial_max_depth
    while depth <= final_max_depth:
        clf = DecisionTreeClassifier(max_depth = depth)
        clf = clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        score_history.append(score)

        depth += depth_increment

    plt.scatter(np.arange(initial_max_depth, final_max_depth + 1, depth_increment), score_history)
    score_history = np.sort(score_history)
    print(score_history)
    plt.show()

from sklearn.ensemble import RandomForestClassifier
def random_forrest(data):
    features = data[:,:-1]
    target =  data[:,-1:]
    X_train, X_test, y_train, y_test = train_test_split (features, target, random_state = 0, test_size = 0.25)

    initial_estimators = 10
    final_estimators = 500
    estimator_increment = 20

    score_history = []
    n_estimators = initial_estimators
    while n_estimators <= final_estimators:
        clf = RandomForestClassifier(n_estimators = n_estimators)
        clf = clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        score_history.append(score)

        n_estimators += estimator_increment

    plt.scatter(np.arange(initial_estimators, final_estimators + 1, estimator_increment), score_history)
    score_history = np.sort(score_history)
    print(score_history)
    plt.show()

#knn(data, 3, 55, 2) # accuracy = 0.55
#call_ann_classification(data, 4, 50, 4) # accuracy = 0.65
#call_svm(data)
#decision_tree(data)
#random_forrest(data)

# binary classification
for i in range(data.shape[0]):
    if (data[i, -1] >= 1):
        data[i, -1] = 1
    else:
        data[i, -1] = 0
call_ann_classification(data, 3, 55, 2) # accuracy = 0.85
