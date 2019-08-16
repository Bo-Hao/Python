import numpy as np
import matplotlib.pyplot as plt
import copy 
from sklearn.cluster import KMeans

class K_means():
    def __init__(self, X, K):
        self.K = K
        self.X = copy.copy(X)


    def fit(self):
        clf = KMeans(n_clusters=self.K)
        clf.fit(self.X)
        self.label = clf.labels_

        return self.label

    def draw(self):
        plt.scatter(self.X[:,0], self.X[:,1], c= self.label)
        plt.title("K Means Cluster")
        plt.show()


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    K = K_means(X, 3)
    res = K.fit()

    K.draw()
