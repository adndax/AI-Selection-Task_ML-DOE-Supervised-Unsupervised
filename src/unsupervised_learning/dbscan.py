import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin

class DBSCAN(BaseEstimator, ClusterMixin):
    def __init__(self, epsilon=0.5, min_samples=5, metric='euclidean', p=2):
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None
        self.p = p
    
    def _compute_distance(self, point, points_array):
        # metric distances
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((points_array - point)**2, axis=1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(points_array - point), axis=1)
        elif self.metric == 'minkowski':
            return np.power(np.sum(np.power(np.abs(points_array - point), self.p), axis=1), 1/self.p)
    
    def fit(self, X, y=None):
        # convert to numpy arra
        X_fit = np.asarray(X)
        
        n_samples = X_fit.shape[0]
        
        # initialize labels array with -1 (noise)
        self.labels_ = np.full(n_samples, -1)
        
        # find core points
        core_points_mask = np.zeros(n_samples, dtype=bool)
        # store neighborhood indices for each point
        neighborhoods = [] 
        
        for i in range(n_samples):
            distances = self._compute_distance(X_fit[i], X_fit)
            neighbors = np.where(distances <= self.epsilon)[0]
            neighborhoods.append(neighbors)
            
            # a point is a core point if it has at least min_samples neighbors (including itself)
            if len(neighbors) >= self.min_samples:
                core_points_mask[i] = True
        
        # start clustering
        cluster_id = 0
        
        # process each core point
        for i in range(n_samples):
            # skip if point is not a core point or already assigned
            if not core_points_mask[i] or self.labels_[i] != -1:
                continue
            
            # start a new cluster
            self.labels_[i] = cluster_id
            
            # process neighborhood (BFS)
            seed_queue = neighborhoods[i].copy().tolist()
            seed_index = 0
            
            while seed_index < len(seed_queue):
                current_point = seed_queue[seed_index]
                
                # if noise point, assign to current cluster
                if self.labels_[current_point] == -1:
                    self.labels_[current_point] = cluster_id
                    
                    # if core point, add its unvisited neighbors to the queue
                    if core_points_mask[current_point]:
                        for neighbor in neighborhoods[current_point]:
                            if self.labels_[neighbor] == -1 and neighbor not in seed_queue:
                                seed_queue.append(neighbor)
                
                seed_index += 1
            
            # move to next cluster
            cluster_id += 1
        
        # store number of clusters
        self.n_clusters_ = cluster_id
        
        return self
    
    def fit_predict(self, X, y=None):
        """Perform DBSCAN clustering and return cluster labels."""
        self.fit(X)
        return self.labels_