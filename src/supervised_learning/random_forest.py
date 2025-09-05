import numpy as np
from supervised_learning.cart import CARTClassifier
from scipy.stats import mode

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=100, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
    
    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)
        
        # teset trees list
        self.trees = []
        
        # store unique classes
        self.classes_ = np.unique(y)
        
        # loop over the number of trees
        for _ in range(self.n_estimators):
            # Create a decision tree instance
            tree = CARTClassifier(max_depth=self.max_depth, 
                                 min_samples_split=self.min_samples_split)
            
            # bootstrap sample indices
            indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            
            # get bootstrap samples
            X_sample = X[indices]
            y_sample = y[indices]
            
            # fit the tree
            tree.fit(X_sample, y_sample)
            
            # add the tree to forest
            self.trees.append(tree)
            
        return self
    
    def predict(self, X):
      
        # ensure X is a numpy array
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
        else:
            X = np.array(X)
            
        # initialize predictions array
        predictions = np.zeros((X.shape[0], len(self.trees)), dtype='int64')
        
        # get predictions from each tree
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        
        # get majority vote for each sample
        result, _ = mode(predictions, axis=1)
        return result.flatten()