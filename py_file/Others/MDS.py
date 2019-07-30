import numpy as np
import matplotlib.pyplot as plt

class MyMDS:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, data):
        m, n = data.shape
        dist = np.zeros((m, m))
        dist = [np.sum(np.square(data[i] - data), axix = 1).reshape(1, m) for i in np.arange(m)]
        disti = [np.mean(dist[i, :]) for i in np.arange(m)]
        distj = [np.mean(dist[:, i]) for i in np.arange(m)]
        distij = mp.mean(dist)
        B = [[-0.5 * (dist[i,j] - disti[i] - distj[j] + distij) for i in np.arange(m)] for j in np.arange(m)]
        lamda, V = np.linalg.eigh(B)
        index = np.argsort(-lamda)[:self.n_components]
        diag_lamda = np.sqrt(np.diag(-np.sort(-lamda)[:self.n_components]))
        V_selected = V[:,index]
        Z = V_selected.dot(diag_lamda)
        return Z


