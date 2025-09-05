import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class KMeans(BaseEstimator, ClassifierMixin):
    def __init__(self, K, max_iter=100, init="random"):
        self.K = K
        self.max_iter = max_iter
        self.init = init
        self.labels_ = None 
        self.centroids = None

    def fit(self, X, y=None):
        # convert to numpy array
        X_fit = np.array(X)

        # init centroids
        if self.init == "random":
            indices = np.random.choice(X_fit.shape[0], self.K, replace=False)
            self.centroids = X_fit[indices]
        elif self.init == "k-means++":
            self.centroids = self._init_kmeans_plus(X_fit)

        for _ in range(self.max_iter):
            # assign label to each data point
            self.labels_ = self._assign_labels(X_fit)

            # update centroids
            new_centroids = self._update_centroids(X_fit, self.labels_)
            
            # check for convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids
            
        return self

    def _init_kmeans_plus(self, X):
        """k-means++ initialization"""
        n_samples = X.shape[0]
        centroids = []
        
        # choose first centroid randomly
        first_idx = np.random.randint(0, n_samples)
        centroids.append(X[first_idx])
        
        for _ in range(1, self.K):
            # compute squared distance each point to its nearest centroid
            distances = np.min(
                np.linalg.norm(X[:, np.newaxis] - np.array(centroids), axis=2) ** 2,
                axis=1
            )
            # compute probability that is proportional to squared distance
            probs = distances / distances.sum()
            chosen_idx = np.random.choice(n_samples, p=probs)
            centroids.append(X[chosen_idx])
        
        return np.array(centroids)

    def _assign_labels(self, X):
        if hasattr(X, 'values'):
            X = X.to_numpy()
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        new_centroids = np.zeros((self.K, X.shape[1]))
        for i in range(self.K):
            if np.sum(labels == i) > 0:
                new_centroids[i] = X[labels == i].mean(axis=0)
            else:
                new_centroids[i] = self.centroids[i]
        return new_centroids
    
    def predict(self, X):
        return self._assign_labels(X)
