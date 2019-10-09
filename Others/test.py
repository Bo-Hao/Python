from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from math import *




iris = datasets.load_iris()
X = iris.data
Y = iris.target

from sklearn.decomposition import PCA

print("---"*4, 'PCA', "---"*4)
pca = PCA(n_components=2)
pcapoints = pca.fit(X).transform(X)
print(pcapoints)

'''plt.scatter(pcapoints[:, 0], pcapoints[:, 1], c = Y)
plt.show()
'''
