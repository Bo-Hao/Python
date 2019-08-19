from sklearn import datasets
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)

from sklearn.tree import DecisionTreeClassifier 
tree = DecisionTreeClassifier(criterion = "entropy", max_depth = 3, random_state = 0)
tree.fit(X_train, Y_train)
'''
print(tree.predict(X_test))
print()
print(Y_test)'''


from sklearn.tree import export_graphviz
#export_graphviz(tree, out_file = "tree.gif")



from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion = 'entropy', n_estimators = 10, random_state = 0, n_jobs = 2)
forest.fit(X_train, Y_train)
print(forest.predict(X_test))


