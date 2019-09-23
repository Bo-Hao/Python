import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

class MyMDS:
    def __init__(self, n_components):
        self.n_components = n_components
        self.n_neighbors = 15
        self.type = 'MDS'

    def fit(self, q):
        D = np.asarray(self.n_components)
        D = np.array([[np.linalg.norm(D[i] - D[j]) for j in np.arange(len(D))] for i in np.arange(len(D))])
        if self.type == 'isomap':
            Max = np.max(D)*1000
            n1, n2 = D.shape
            k = self.n_neighbors
            D1 = np.ones((n1, n1)) * Max
            D_arg = np.argsort(D, axis = 1)
            for i in np.arange(n1):
                D1[i, D_arg[i, 0:k+1]] = D[i, D_arg[i, 0:k+1]]
        
            for k in np.arange(n1):
                for i in np.arange(n1):
                    for j in np.arange(n1):
                        D1[i, j] = D1[i, k] + D1[k, j] if D1[i,k] + D1[k,j] < D1[i,j] else D1[i, j]
            D = D1
        D_square = D ** 2
        total_Mean = np.mean(D_square)
        column_Mean = np.mean(D_square, axis = 0)
        row_Mean = np.mean(D_square, axis = 1)
        B1 = [[-0.5 * (D_square[i][j] - row_Mean[i] - column_Mean[j] + total_Mean) 
        for i in np.arange(D_square.shape[0])] for j in np.arange(D_square.shape[1])]
        eigenValue, eigenVector = np.linalg.eig(B1)
        self.Z = np.dot(eigenVector[:, :q], np.sqrt(np.diag(eigenValue[:q])))
    
    def plot(self, tar = ''):
        if self.Z.shape[1] == 2:
            if tar == '':
                plt.scatter(self.Z.T[0], self.Z.T[1])
            else:
                plt.scatter(self.Z.T[0], self.Z.T[1], c = target)
            plt.title('MDS')
            plt.show()
        else:
            print("Not 2D")
        


    
def dist(data):
    data = np.array([[np.linalg.norm(data[i] - data[j]) for j in np.arange(len(data))] for i in np.arange(len(data))])
    return data
        
def generate_curve_data():
    xx,target=datasets.samples_generator.make_s_curve(400, random_state=9)
    return xx,target

def floyd(D,n_neighbors=15):
    Max = np.max(D)*1000
    n1, n2=D.shape
    k = n_neighbors
    D1 = np.ones((n1, n1)) * Max
    D_arg = np.argsort(D, axis = 1)
    for i in np.arange(n1):
        D1[i, D_arg[i, 0:k+1]] = D[i, D_arg[i, 0:k+1]]

    for k in np.arange(n1):
        for i in np.arange(n1):
            for j in np.arange(n1):
                D1[i, j] = D1[i, k] + D1[k, j] if D1[i,k] + D1[k,j] < D1[i,j] else D1[i, j]
                
    return D1




        
        
        
        
        
        
        
        
        
        
        