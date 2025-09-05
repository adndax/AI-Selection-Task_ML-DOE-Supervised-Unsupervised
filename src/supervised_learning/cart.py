import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# tree node
class TreeNode:
    def __init__(self, data, feature_idx=None, feature_val=None, label=None):
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.label = label  # store labels in leaf classes
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None and self.right is None

class CARTClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    # gini
    def gini(self, probs):
        return 1.0 - np.sum(probs ** 2)

    def class_probabilities(self, labels):
        counts = np.array([np.sum(labels == c) for c in self.labels_in_train])
        total = np.sum(counts)
        return counts / total if total > 0 else np.zeros(len(self.labels_in_train))

    def data_gini(self, labels):
        return self.gini(self.class_probabilities(labels))

    def partition_gini(self, subsets):
        total_count = sum(len(s) for s in subsets)
        return sum((len(s)/total_count) * self.data_gini(s) for s in subsets if len(s) > 0)

    # code for splitting
    def split(self, data, feature_idx, feature_val):
        left = data[data[:, feature_idx] <= feature_val]
        right = data[data[:, feature_idx] > feature_val]
        return left, right

    # finding best split (feature and its threshold) based on lowest gini
    def find_best_split(self, data):
        min_gini = float('inf')
        best_idx, best_val = None, None
        best_splits = (None, None)

        n_features = data.shape[1] - 1
        feature_indices = np.arange(n_features)
        if self.max_features and self.max_features < n_features:
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)

        for idx in feature_indices:
            values = np.unique(data[:, idx])
            if len(values) == 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2
            for val in thresholds:
                left, right = self.split(data, idx, val)
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue
                g = self.partition_gini([left[:, -1], right[:, -1]])
                if g < min_gini:
                    min_gini = g
                    best_idx, best_val = idx, val
                    best_splits = (left, right)
        return best_splits[0], best_splits[1], best_idx, best_val

    # tree construction
    def create_tree(self, data, depth=0):
        labels = data[:, -1]
    
        majority_label = self.labels_in_train[np.argmax(self.class_probabilities(labels))]
        if (self.max_depth and depth >= self.max_depth) or len(data) < self.min_samples_split:
            return TreeNode(data, label=majority_label)

        left, right, idx, val = self.find_best_split(data)
        if idx is None:
            return TreeNode(data, label=majority_label)

        node = TreeNode(data, feature_idx=idx, feature_val=val)
        node.left = self.create_tree(left, depth+1)
        node.right = self.create_tree(right, depth+1)
        return node

    # prediction for each sample (from root to node)
    def predict_one_sample(self, x):
        node = self.tree_
        while not node.is_leaf():
            if x[node.feature_idx] <= node.feature_val:
                node = node.left
            else:
                node = node.right
        return node.label  

    def fit(self, X, y):
        self.labels_in_train = np.unique(y)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        train_data = np.concatenate((X, y), axis=1)
        self.tree_ = self.create_tree(train_data)
        return self

    # final prediction  
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self.predict_one_sample(x) for x in X])
