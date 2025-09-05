import pandas as pd
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return np.where(z > 0, 1, 0)

def linear(z):
    return z

def linear_deriv(z):
    return np.ones_like(z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

class DenseLayer:
    def __init__(self, neurons, input_size, activation='relu', init_method='he'):
        if init_method == 'xavier':
            self.weights = np.random.randn(neurons, input_size) * np.sqrt(1 / input_size)
        elif init_method == 'he':
            self.weights = np.random.randn(neurons, input_size) * np.sqrt(2 / input_size)
        elif init_method == 'random':
            self.weights = np.random.randn(neurons, input_size) * 0.01
        else:
            raise ValueError("Unsupported init_method")
        self.biases = np.zeros((neurons, 1))
        self.activation = activation
        self.z = None
        self.a = None
        self.input = None

    def forward(self, input_data):
        self.input = input_data  # shape: (n_samples, n_features)
        self.z = np.dot(self.input, self.weights.T) + self.biases.T  # shape: (n_samples, neurons)
        if self.activation == 'sigmoid':
            self.a = sigmoid(self.z)
        elif self.activation == 'relu':
            self.a = relu(self.z)
        elif self.activation == 'linear':
            self.a = linear(self.z)
        elif self.activation == 'softmax':
            self.a = softmax(self.z)
        else:
            raise ValueError("Unsupported activation")
        return self.a

    def backward(self, dA, is_output=False, loss_type=None, y=None):
        if is_output and loss_type == 'binary_cross_entropy' and self.activation == 'sigmoid':
            dZ = self.a - y  # shape: (n_samples, 1)
        elif self.activation == 'sigmoid':
            dZ = dA * sigmoid_deriv(self.a)
        elif self.activation == 'relu':
            dZ = dA * relu_deriv(self.z)
        elif self.activation == 'linear':
            dZ = dA * linear_deriv(self.z)
        elif self.activation == 'softmax':
            raise ValueError("Softmax backward only supported with cross-entropy")
        else:
            raise ValueError("Unsupported activation")
        dW = np.dot(dZ.T, self.input) / self.input.shape[0]  # shape: (neurons, n_features)
        db = np.sum(dZ, axis=0, keepdims=True).T / self.input.shape[0]  # shape: (neurons, 1)
        dInput = np.dot(dZ, self.weights)  # shape: (n_samples, n_features)
        return dW, db, dInput

class ANN:
    def __init__(self, layer_sizes, activations, init_methods, loss='mse', regularization=None, lambda_reg=0.01, 
                 optimizer='gd', lr=0.01, optimizer_params=None):
        if len(layer_sizes) < 2:
            raise ValueError("At least input and output sizes required")
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Activations must be provided for each hidden and output layer")
        if len(init_methods) != len(layer_sizes) - 1:
            raise ValueError("Init methods must be provided for each hidden and output layer")
        
        self.layers = []
        for i in range(1, len(layer_sizes)):
            layer = DenseLayer(layer_sizes[i], layer_sizes[i-1], activations[i-1], init_methods[i-1])
            self.layers.append(layer)
        
        self.loss_type = loss
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.optimizer = optimizer
        self.lr = lr
        self.optimizer_params = optimizer_params or {}
        
        if self.optimizer == 'adam':
            self.beta1 = self.optimizer_params.get('beta1', 0.9)
            self.beta2 = self.optimizer_params.get('beta2', 0.999)
            self.epsilon = self.optimizer_params.get('epsilon', 1e-8)
            self.t = 0
            self.m_weights = [np.zeros_like(layer.weights) for layer in self.layers]
            self.v_weights = [np.zeros_like(layer.weights) for layer in self.layers]
            self.m_biases = [np.zeros_like(layer.biases) for layer in self.layers]
            self.v_biases = [np.zeros_like(layer.biases) for layer in self.layers]
        elif self.optimizer != 'gd':
            raise ValueError("Unsupported optimizer")

    def forward_prop(self, X):
        a = X  # shape: (n_samples, n_features)
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        if self.loss_type == 'mse':
            loss = np.sum((y_pred - y_true) ** 2) / (2 * m)
        elif self.loss_type == 'binary_cross_entropy':
            loss = -np.sum(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8)) / m
        else:
            raise ValueError("Unsupported loss")
        
        reg_term = 0
        if self.regularization == 'l1':
            reg_term = self.lambda_reg * sum(np.sum(np.abs(layer.weights)) for layer in self.layers) / m
        elif self.regularization == 'l2':
            reg_term = (self.lambda_reg / (2 * m)) * sum(np.sum(layer.weights ** 2) for layer in self.layers)
        
        return loss + reg_term

    def backward_prop(self, y_pred, y_true):
        m = y_true.shape[0]
        if self.loss_type == 'mse':
            dA = (y_pred - y_true) / m
        elif self.loss_type == 'binary_cross_entropy':
            dA = (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-8) / m
        else:
            raise ValueError("Unsupported loss")
        
        grads = []
        for i, layer in enumerate(reversed(self.layers)):
            is_output = (i == 0)
            dW, db, dA = layer.backward(dA, is_output, self.loss_type, y_true if is_output else None)
            
            if self.regularization == 'l1':
                dW += (self.lambda_reg / m) * np.sign(layer.weights)
            elif self.regularization == 'l2':
                dW += (self.lambda_reg / m) * layer.weights
            
            grads.append((dW, db))
        
        return list(reversed(grads))

    def update_weights(self, grads):
        if self.optimizer == 'gd':
            for i, (dW, db) in enumerate(grads):
                self.layers[i].weights -= self.lr * dW
                self.layers[i].biases -= self.lr * db
        elif self.optimizer == 'adam':
            self.t += 1
            for i, (dW, db) in enumerate(grads):
                self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * dW
                self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (dW ** 2)
                m_hat_w = self.m_weights[i] / (1 - self.beta1 ** self.t)
                v_hat_w = self.v_weights[i] / (1 - self.beta2 ** self.t)
                self.layers[i].weights -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                
                self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * db
                self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (db ** 2)
                m_hat_b = self.m_biases[i] / (1 - self.beta1 ** self.t)
                v_hat_b = self.v_biases[i] / (1 - self.beta2 ** self.t)
                self.layers[i].biases -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    def fit(self, X, y, epochs, batch_size):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)  # shape: (n_samples, 1)
        
        if X.shape[1] != self.layers[0].weights.shape[1]:
            raise ValueError(f"Input shape mismatch: Expected {self.layers[0].weights.shape[1]} features, got {X.shape[1]}")
        if y.shape[1] != self.layers[-1].biases.shape[0]:
            raise ValueError(f"Output shape mismatch: Expected {self.layers[-1].biases.shape[0]} outputs, got {y.shape[1]}")
        
        print(f"X shape in fit: {X.shape}")
        print(f"y shape in fit: {y.shape}")
        
        m = X.shape[0]
        losses = []
        for epoch in range(epochs):
            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                X_batch = X[start:end]
                y_batch = y[start:end]
                y_pred = self.forward_prop(X_batch)
                loss = self.compute_loss(y_pred, y_batch)
                grads = self.backward_prop(y_pred, y_batch)
                self.update_weights(grads)
            losses.append(loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        return losses

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        print(f"X shape in predict: {X.shape}")
        y_pred = self.forward_prop(X)
        print(f"y_pred shape: {y_pred.shape}")
        return y_pred