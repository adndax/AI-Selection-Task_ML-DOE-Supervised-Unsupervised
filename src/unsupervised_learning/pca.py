import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        if (self.n_components ==  None):
            n_samples = X.shape[0]
            n_features = X.shape[1]
            self.n_components = min(n_samples, n_features)

        features = X.T
        # store the covariance matrix
        cov_matrix = np.cov(features)

        # Eigen Decomposition and Identifying Principal Components
        self.eig_values_, self.eig_vectors_ = np.linalg.eig(cov_matrix)
        indices_sorted_eig_val = np.argsort(self.eig_values_)[::-1]
        self.eig_values_ = self.eig_values_[indices_sorted_eig_val]
        self.eig_vectors_ = self.eig_vectors_[:, indices_sorted_eig_val]

        # explained variance is the first n eigen values
        self.explained_variance_ = self.eig_values_[ :self.n_components]
        return self

    def transform(self, X, y=None):
        feature_vector = self.eig_vectors_[:, :self.n_components]
        X_transform = np.dot(X, feature_vector)

        return pd.DataFrame(X_transform)

    def explained_variance(self):
        return self.explained_variance_
    
    def explained_variance_print(self):
        total_eig_val = np.sum(self.eig_values_)
        for i in range(self.n_components):
            ind_var = (self.eig_values_[i]*100 / total_eig_val)
            cum_var = np.sum(self.eig_values_[:i+1]*100 / total_eig_val)
            print(f'{i+1}th eigen vectors: {self.eig_vectors_[:, i]}')
            print(f'Explained Variance {ind_var:.2f} %')
            print(f'Cumulaative Explained Variance: {cum_var:.2f} %\n')

    


