import numpy as np
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin

class KNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, K, distance):
        self.K = K
        self.distance = distance

    def euclidean(self, x, x_train):
        return np.sqrt(np.sum(np.square(x_train - x), axis=1))
    
    def manhattan(self, x, x_train):
        return np.sum(np.abs(x_train - x), axis=1)
    
    def minkowski(self, x, x_train, p):
        return np.power(np.sum(np.power(np.abs(x_train - x), p), axis= 1), 1/p)

    def find_neighbors(self, x):
        if (self.distance == 'euclidean'):
            distances = self.euclidean(x, self.X_train)
        elif (self.distance == 'manhattan'):
            distances = self.manhattan(x, self.X_train)
        elif (self.distance == 'minkowski'):
            distances = self.minkowski(x, self.X_train)

        sorted_indices = distances.argsort()
        k_nearest_indices = sorted_indices[:self.K]
        return self.y_train.iloc[k_nearest_indices]
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
        return self

    def predict(self, X):
        m = X.shape[0]

        y_pred = np.zeros(m)
        for i in range(m):
            x = X.iloc[i]
            neighbors = self.find_neighbors(x)
            y_pred[i] = mode(neighbors)[0]
        
        return y_pred