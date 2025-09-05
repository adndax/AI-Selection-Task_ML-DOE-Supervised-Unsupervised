import numpy as np
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator, ClassifierMixin

def linear_kernel(X1, X2):
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    return np.dot(X1, X2.T)


def rbf_kernel(X1, X2, gamma=0.5):
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    diff = X1[:, None, :] - X2[None, :, :]
    return np.exp(-gamma * np.sum(diff ** 2, axis=2))

class SVC(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel="rbf", gamma=0.5):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.w = None
        self.b = None
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None

    def fit(self, X, y):
        '''Using Quadratic Programming'''

        X = np.asarray(X)
        y = np.asarray(y)

        # make sure that labels in y is either -1 and 1
        y = np.where(y == 0, -1, 1)

        n_samples, n_features = X.shape

        # Pilih kernel
        if self.kernel == "linear":
            K = linear_kernel(X, X)
        elif self.kernel == "rbf":
            K = rbf_kernel(X, X, self.gamma)
        else:
            raise ValueError(f"Kernel '{self.kernel}' tidak dikenal")

        # Setup Quadratic Programming:
        # minimize (1/2) αᵀ P α + qᵀ α
        P = matrix(np.outer(y, y) * K, tc="d")
        q = matrix(-np.ones(n_samples), tc="d")

        # inequality constraints: 0 <= αᵢ <= C
        G_std = np.diag(-np.ones(n_samples))
        h_std = np.zeros(n_samples)
        G_slack = np.diag(np.ones(n_samples))
        h_slack = np.ones(n_samples) * self.C
        G = matrix(np.vstack((G_std, G_slack)), tc="d")
        h = matrix(np.hstack((h_std, h_slack)), tc="d")

        # equality constraint: Σ αᵢ yᵢ = 0
        A = matrix(y.reshape(1, -1), tc="d")
        b = matrix(0.0)

        # solve QP
        solvers.options["show_progress"] = False
        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(sol["x"])

        # storing support vectors
        sv = alphas > 1e-5
        self.support_vectors = X[sv]
        self.support_vector_labels = y[sv]
        self.alphas = alphas[sv]

        # compute w only for linear kernel
        if self.kernel == "linear":
            self.w = np.sum(
                (self.alphas * self.support_vector_labels)[:, None]
                * self.support_vectors,
                axis=0,
            )

        # compute bias 
        self.b = np.mean([
            y_k
            - np.sum(
                self.alphas * self.support_vector_labels *
                (
                    linear_kernel(self.support_vectors, x_k[None])
                    if self.kernel == "linear"
                    else rbf_kernel(self.support_vectors, x_k[None], self.gamma)
                ).ravel()
            )
            for x_k, y_k in zip(self.support_vectors, self.support_vector_labels)
        ])

        return self

    def project(self, X):
        # Compute Decision Function f(x)
        X = np.asarray(X)
        if self.kernel == "linear":
            return np.dot(X, self.w) + self.b
        else:  # RBF kernel
            return (self.alphas * self.support_vector_labels) @ \
                   rbf_kernel(self.support_vectors, X, self.gamma) + self.b

    def predict(self, X):
    
        decision = self.project(X)
        preds = np.sign(decision)
        
        # for binary classification, convert result from {-1, 1} to {0, 1}
        return np.where(preds == -1, 0, 1)
