import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, scaler = StandardScaler()):
        self.scaler = scaler

    def fit(self, X, y=None):
        X_fit = X.copy()
        
        self.ordinal_features_ = ['slope', 'thal', 'ca']
        self.nominal_features_ = ['sex', 'restecg', 'cp', 'fbs', 'exang']
        self.numerical_features_ = [col for col in X_fit.select_dtypes(include=np.number).columns if col not in self.ordinal_features_ and col not in self.nominal_features_]
        
        likely_missing_features = ['trestbps', 'chol']
        for col in likely_missing_features:
            X_fit[col] = X_fit[col].replace(0, np.nan)
        
        ordinal_pipeline = Pipeline([
            ('encoder', OrdinalEncoder(
                categories=[
                    ['downsloping', 'flat', 'upsloping'],
                    ['normal', 'fixed defect', 'reversible defect'],
                    [0, 1, 2, 3]
                ],
                handle_unknown='use_encoded_value',
                unknown_value=np.nan
            )),
            ('imputer', IterativeImputer(initial_strategy='mean'))
        ])
    
        nominal_pipeline = Pipeline([
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ]) 

        numeric_pipeline = Pipeline([
            ('imputer', IterativeImputer(initial_strategy='mean')),
        ])

        self.preprocessor_ = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, self.numerical_features_),
                ('ord', ordinal_pipeline, self.ordinal_features_),
                ('nom', nominal_pipeline, self.nominal_features_)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False,
        )
        
        X_fit = self.preprocessor_.fit_transform(X_fit)
        feature_names = self.preprocessor_.get_feature_names_out()
        X_fit = pd.DataFrame(X_fit, columns=feature_names)
        

        nominal_nan_col = [x for x in X_fit.columns if '_nan' in x]
        prefix_nan = [x.split('_')[0] for x in nominal_nan_col]
        nominal_with_nan = [x for x in X_fit.columns if x.split('_')[0] in prefix_nan and x.split('_')[-1] != 'nan']

        for nominal_nan in nominal_nan_col:
            prefix_nan = nominal_nan.split('_')[0] 
            for nominal in nominal_with_nan:
                if nominal in X_fit.columns:
                    prefix = nominal.split('_')[0]
                    if prefix_nan == prefix:
                        X_fit.loc[X_fit[nominal_nan] == True, nominal] = np.nan

        self.nominal_imputer_ = IterativeImputer()
        self.nominal_imputer_.fit(X_fit)
        X_fit.drop(nominal_nan_col, axis=1, inplace=True)
        self.scaler.fit(X_fit)

        return self

    def transform(self, X, y=None):
        X_transform = X.copy()
        
        likely_missing_features = ['trestbps', 'chol']
        for col in likely_missing_features:
            X_transform[col] = X_transform[col].replace(0, np.nan)

        X_transform = self.preprocessor_.transform(X_transform)
        feature_names = self.preprocessor_.get_feature_names_out()
        X_transform = pd.DataFrame(X_transform, columns=feature_names)

        nominal_nan_col = [x for x in X_transform.columns if '_nan' in x]
        prefix_nan = [x.split('_')[0] for x in nominal_nan_col]
        nominal_with_nan = [x for x in X_transform.columns if x.split('_')[0] in prefix_nan and x.split('_')[-1] != 'nan']
        
        for nominal_nan in nominal_nan_col:
            prefix_nan = nominal_nan.split('_')[0] 
            for nominal in nominal_with_nan:
                if nominal in X_transform.columns:
                    prefix = nominal.split('_')[0]
                    if prefix_nan == prefix:
                        X_transform.loc[X_transform[nominal_nan] == True, nominal] = np.nan

        X_transform_columns = X_transform.columns
        X_transform = self.nominal_imputer_.transform(X_transform)
        
        X_transform = pd.DataFrame(X_transform, columns=X_transform_columns).drop(nominal_nan_col, axis=1)

        X_transform_columns = X_transform.columns
        X_transform = self.scaler.transform(X_transform)

        X_transform = pd.DataFrame(X_transform, columns=X_transform_columns)

        categorical_features = nominal_with_nan + self.ordinal_features_ + [x for x in X_transform.columns if x not in self.ordinal_features_ and x not in self.numerical_features_]

        for col in categorical_features:
            X_transform[col] = X_transform[col].round().astype(int)

        

        return X_transform
 