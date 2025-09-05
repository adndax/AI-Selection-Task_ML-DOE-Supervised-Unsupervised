import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class GaussianNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, var_smoothing=1e-09):
        self.alpha = alpha
        self.var_smoothing = var_smoothing
    
    def fit(self, X, y):

        # store each class
        self.classes_ = np.unique(y)

        # splitting features by data type for different handling probability
        self.continuous_features_ = X.select_dtypes(include=['float64']).columns.tolist()
        self.discrete_features_ = X.select_dtypes(include=['int64']).columns.tolist()

        # compute priors probability for each class
        self.priors_ = y.value_counts(normalize=True)
        
        # merge x and y for easier analysis
        y.name = 'has_disease'
        df_train = X.join(y)

        # for continuous feature, compute mean and std 
        self.means_ = df_train.groupby('has_disease')[self.continuous_features_].mean()
        self.stds_ = df_train.groupby('has_disease')[self.continuous_features_].std()

        # compute likelihood for discrete feature
        self.discrete_likelihoods_ = {}
        self.feature_vocabs_ = {f: len(X[f].unique()) for f in self.discrete_features_}
        
        for feature in self.discrete_features_:
            # compute probability for each combination of (class, feature_value)
            likelihood_dict = {}
            
            for c in self.classes_:
                # data for current class
                X_class = df_train[df_train['has_disease'] == c]
                n_class = len(X_class)
                
                value_counts = X_class[feature].value_counts()
                vocab_size = self.feature_vocabs_[feature]
                
                # apply laplace smoothing
                probs = {}
                for value in X[feature].unique():
                    count = value_counts.get(value, 0)
                    prob = (count + self.alpha) / (n_class + self.alpha * vocab_size)
                    probs[value] = prob
                
                likelihood_dict[c] = probs
            
            self.discrete_likelihoods_[feature] = likelihood_dict

        return self

    def _pdf(self, X_continuous, class_idx):
        # computing gaussian pdf
        mean = self.means_.loc[class_idx]
        std = self.stds_.loc[class_idx]
        
        numerator = np.exp(-((X_continuous - mean)**2) / (2 * (std**2 + self.var_smoothing)))
        denominator = np.sqrt(2 * np.pi * (std**2 + self.var_smoothing))
        
        return numerator / denominator

    def predict(self, X):
        n_samples = X.shape[0]  

        # log is used for computational stability
        log_posteriors = np.zeros((n_samples, len(self.classes_)))

        for idx, c in enumerate(self.classes_):
        # log posterior
            log_posterior = np.log(self.priors_[c]) * np.ones(n_samples)
    
        # log of likelihood for continuous feature
            if self.continuous_features_:
                X_continuous = X[self.continuous_features_]
                pdf_values = self._pdf(X_continuous, c)
                # Handle zero probabilities
                pdf_values = np.maximum(pdf_values, 1e-300)
                log_posterior += np.log(pdf_values).sum(axis=1)

        # log of likelihood for discrete feature
            for feature in self.discrete_features_:

                if feature not in self.discrete_likelihoods_:
                    continue

                feature_likelihoods = self.discrete_likelihoods_[feature][c]
                
                # map values to probabilities
                probs = X[feature].map(feature_likelihoods)
                
                # handle unseen values
                vocab_size = self.feature_vocabs_[feature]
                n_class = self.priors_[c] * len(X)  
                unseen_prob = self.alpha / (n_class + self.alpha * vocab_size)
                probs = probs.fillna(unseen_prob)
                
                # add log probability
                log_posterior += np.log(np.maximum(probs, 1e-300))

            log_posteriors[:, idx] = log_posterior

        # return class with highest log posterior
        return self.classes_[np.argmax(log_posteriors, axis=1)]