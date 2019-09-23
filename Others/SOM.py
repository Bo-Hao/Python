import random
import numpy as np
import copy
import math
import matplotlib.pyplot as plt


class SOM:
    def __init__(self, dataset):
        self.data = copy.copy(dataset)
        self.T = 50 
        
        
        
    def init(self, n, m):
        def cal2NF(X):#計算向量的二範數
            res = 0
            for x in X:
                res += x**2
            return res ** 0.5
        #學習率函式
        def eta(t,N):
            return (0.3/(t+1))* (math.e ** -N)
        
        self.n = n
        self.m = m
        d = len(self.data[0])
        self.d = d
        size = n * m * d
        array = np.array([random.random() for i in range(size)])
        self.com_weight = array.reshape(n,m,self.d)
        self.N_neibor = 5
        
        #對資料集進行歸一化處理
        self.old_dataSet = copy.copy(self.data)
        for data in self.data:
            two_NF = cal2NF(data)
            for i in range(len(data)):
                data[i] = data[i] / two_NF
        
        #對權值矩陣進行歸一化處理
        for x in self.com_weight:
            for data in x:
                two_NF = cal2NF(data)
                for i in range(len(data)):
                    data[i] = data[i] / two_NF
        
                    
    #SOM演算法的實現
    def fit(self):
        #得到獲勝神經元的索引值
        def getWinner(data, com_weight):
            max_sim = 0
            mark_n = 0
            mark_m = 0
            for i in range(self.n):
                for j in range(self.m):
                    if sum(data * self.com_weight[i, j]) > max_sim:
                        max_sim = sum(data * com_weight[i,j])
                        mark_n = i
                        mark_m = j
            return mark_n , mark_m
        
        #得到神經元的N鄰域
        def getNeibor(n , m, N_neibor , com_weight):
            res = []
            nn,mm , _ = np.array(com_weight).shape
            for i in range(nn):
                for j in range(mm):
                    N = int(((i-n)**2+(j-m)**2)**0.5)
                    if N<=N_neibor:
                        res.append((i,j,N))
            return res

        #學習率函式
        def eta(t,N):
            return (0.3/(t+1))* (math.e ** -N)
                
        for t in range(self.T-1):
            for data in self.data:
                n , m = getWinner(data, self.com_weight)
                neibor = getNeibor(self.n , self.m , self.N_neibor , self.com_weight)
                for x in neibor:
                    j_n=x[0];j_m=x[1];N=x[2]
                    #權值調整
                    self.com_weight[j_n][j_m] = self.com_weight[j_n][j_m] + eta(t,N)*(data - self.com_weight[j_n][j_m])
                self.N_neibor = self.N_neibor+1-(t+1)/200
        res = {}
        N , M , _ = np.array(self.com_weight).shape
        for i in range(len(self.data)):
            n, m = getWinner(self.data[i], self.com_weight)
            key = n*M + m
            if key in res:
                res[key].append(i)
            else:
                res[key] = []
                res[key].append(i)
        self.res = res
        return res
        
    def draw(self):
        color = ['red', 'yellow', 'green', 'blue', 'pink', 'gray', 'purple' , 'navy']
        count = 0
        for i in self.res.keys():
            X = []
            Y = []
            datas = self.res[i]
            for j in range(len(datas)):
                X.append(self.data[datas[j]][0])
                Y.append(self.data[datas[j]][1])
            plt.scatter(X, Y, marker='o', c=color[count % len(color)], label=i)
            count += 1
        #plt.legend(loc='upper right')
        plt.title('SOM')
        plt.show()

    def drawbytarget(self, target):
        x = self.data[:, 0]
        y = self.data[:, 1]
        plt.scatter(x, y, c= target, marker = 'o', label = target)
        plt.title("By target")
        plt.show()




if __name__ == '__main__':
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    
    
    print("---"*4, 'MDS', "---"*4)
    from sklearn.manifold import MDS
    embedding = MDS(n_components=2)
    X_transformed = embedding.fit_transform(X)
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c = Y)
    plt.show()
    from sklearn.decomposition import PCA
    
    print("---"*4, 'PCA', "---"*4)
    pca = PCA(n_components=2)
    pcapoints = pca.fit(X).transform(X)
    plt.scatter(pcapoints[:, 0], pcapoints[:, 1], c = Y)
    plt.show()
    
    print("---"*4, "SOM", "---"*4)
    S = SOM(X)
    S.init(10, 10)
    res = S.fit()
    S.draw()
    S.drawbytarget(Y)

    