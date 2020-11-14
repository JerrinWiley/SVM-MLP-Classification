from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import KFold
data = pd.read_csv("ionosphere.data", header=None)
kf = KFold(n_splits=10)
X = data.iloc[:,0:-1].values
y = data.iloc[:,-1].values

# MLPClassifier using Relu activation
print("MLP classification using Relu activation (10-fold cross validation scores):")
clf_relu = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=3000,activation = 'relu',solver='adam',random_state=1)
for train_indices, test_indices in kf.split(X):
    clf_relu.fit(X[train_indices], y[train_indices])
    print(clf_relu.score(X[test_indices], y[test_indices]))

print("MLP classification using Tanh activation (10-fold cross validation scores):")
clf__tanh = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=3000,activation = 'tanh',solver='adam',random_state=12)
for train_indices, test_indices in kf.split(X):
    clf__tanh.fit(X[train_indices], y[train_indices])
    print(clf__tanh.score(X[test_indices], y[test_indices]))

print("SVM classification using Linear kernel (10-fold cross validation scores):")
clf__linear = svclassifier_linear = SVC(kernel='linear')
for train_indices, test_indices in kf.split(X):
    clf__linear.fit(X[train_indices], y[train_indices])
    print(clf__linear.score(X[test_indices], y[test_indices]))

print("SVM classification using Guassian kernel (10-fold cross validation scores):")
clf__rbf = svclassifier_linear = SVC(kernel='rbf')
for train_indices, test_indices in kf.split(X):
    clf__rbf.fit(X[train_indices], y[train_indices])
    print(clf__rbf.score(X[test_indices], y[test_indices]))

