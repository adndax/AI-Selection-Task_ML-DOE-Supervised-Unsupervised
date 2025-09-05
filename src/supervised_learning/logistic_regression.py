import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class LogisticRegressionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.01, n_iters=1000, regularization_term=None, lambda_reg=0.01, optimizer='gradient_descent'):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.regularization_term = regularization_term
        self.lambda_reg = lambda_reg
        self.optimizer = optimizer
        self.weights_ = None
        self.bias_ = None
        self.losses = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y_pred, y):
        # y : true class
        # y_pred : model prediction

        # cross-Entropy
        epsilon = 1e-9
        n = len(y)
        ce_loss = -(1/n) * np.sum(y*np.log(y_pred + epsilon) + (1-y)*np.log(1-y_pred + epsilon))

        # regularization
        if self.regularization_term == 'l1':
            reg_penalty = self.lambda_reg * np.sum(np.abs(self.weights_))
        elif self.regularization_term == 'l2':
            reg_penalty = self.lambda_reg * np.sum(self.weights_**2) / 2
        else: 
            reg_penalty = 0

        return ce_loss + reg_penalty
    
    def _gradient_step(self, X, y, y_pred):
        m = X.shape[0]
        
        # compute Gradients
        dz = y_pred - y
        dw = (1/m) * np.dot(X.T, dz)
        db = (1/m) * np.sum(dz)
        
        # add regularization
        if self.regularization_term == 'l1':
            dw += self.lambda_reg * np.sign(self.weights_)
        elif self.regularization_term == 'l2':
            dw += self.lambda_reg * self.weights_
        
        # update weights and bias
        self.weights_ -= self.learning_rate * dw
        self.bias_ -= self.learning_rate * db
    
    def _newton_step(self, X, y, y_pred):
        m, n = X.shape
        
        # gradient
        dz = y_pred - y
        gradient = (1/m) * np.dot(X.T, dz)
        
        # Hessian
        S = y_pred * (1 - y_pred)
        hessian = (1/m) * np.dot(X.T * S, X)
        
        # add regularization
        if self.regularization_term == 'l2':
            gradient += self.lambda_reg * self.weights_
            hessian += self.lambda_reg * np.eye(n)
        elif self.regularization_term == 'l1':
            gradient += self.lambda_reg * np.sign(self.weights_)
            hessian += 0.01 * self.lambda_reg * np.eye(n)  # Approximation untuk L1
        
        # stabilize Hessian
        hessian += 1e-8 * np.eye(n)
        
        # Newton update
        delta_w, _, _, _ = np.linalg.lstsq(hessian, gradient, rcond=None)
        self.weights_ -= delta_w
        
        # update bias
        db = (1/m) * np.sum(dz)
        self.bias_ -= db
    
    def fit(self, X, y):
        n = X.shape[1]

        # initiate parameters weights and bias = 0
        self.weights_ = np.zeros(n)
        self.bias_ = 0

        for _ in range(self.n_iters):
            z = np.dot(X, self.weights_) + self.bias_
            y_pred = self._sigmoid(z)
            
            # Choose optimizer
            if self.optimizer == 'newton':
                self._newton_step(X, y, y_pred)
            else:
                self._gradient_step(X, y, y_pred)
            
            self.losses.append(self.compute_loss(y_pred, y))

    def predict(self, X):
        threshold = 0.5
        z = np.dot(X, self.weights_) + self.bias_
        y_pred_proba = self._sigmoid(z)
        y_pred_cls = (y_pred_proba > threshold).astype(int)
        
        return y_pred_cls
