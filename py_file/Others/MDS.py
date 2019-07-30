import numpy as np
import matplotlib.pyplot as plt


class MyMDS:
    def __init__(self, n_components):
        self.n_components = n_components
    
    def fit(self, q):
        D = np.asarray(self.n_components)
        D_square = D ** 2
        total_Mean = np.mean(D_square)
        column_Mean = np.mean(D_square, axis = 0)
        row_Mean = np.mean(D_square, axis = 1)
        B = [[-0.5 * (D_square[i][j] - row_Mean[i] - column_Mean[j] + total_Mean) 
        for i in np.arange(D_square.shape[0])] for j in np.arange(D_square.shape[1])]
        eigenValue, eigenVector = np.linalg.eig(B)
        self.Z = np.dot(eigenVector[:, :q], np.sqrt(np.diag(eigenValue[:q])))
    
    def plot2D(self):
        if self.Z.shape[1] == 2:
            for i in np.arange(self.Z.shape[0]):
                plt.scatter(self.Z[i][0], self.Z[i][1])
            
            plt.show()
        else:
            print("Not 2D")
        
        

    