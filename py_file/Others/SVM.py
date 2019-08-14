from sklearn import svm 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class SVM:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.mode = 'linear'
        self.S = svm.SVC(kernel = self.mode)
    
    def fit(self):
        self.S.fit(self.X, self.Y)
        
        #print(self.S) 
        #print(self.S.support_vectors_) #支援向量點 
        #print(self.S.support_) #支援向量點的索引 
        #print(self.S.n_support_) #每個class有幾個支援向量點 
        
    def predict(self, x):
        self.x = x
        self.prediction = self.S.predict(x)
        
    def draw(self):
        plt.scatter(self.x[:, 0], self.x[:, 1], c = self.prediction)
        plt.show()


if __name__ == '__main__':
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    X, Y = X_test, y_test
    
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
    
    print("----"*3, "SVM", "----"*3)
    S = SVM(X_train, y_train)
    S.fit()
    S.predict(X)
    S.draw()