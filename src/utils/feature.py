from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, verbose=True):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.verbose = verbose
        self.selector = None
        self.feature_names_ = None
        self.importances_ = None
        self.sorted_indices_ = None

    def fit(self, X, y):
        if self.verbose:
            print("🎯 Fitting RandomForest for Feature Selection...")
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X, y)

        self.selector = SelectFromModel(rf, prefit=True)
        self.feature_names_ = X.columns[self.selector.get_support()]

        self.importances_ = rf.feature_importances_
        self.sorted_indices_ = np.argsort(self.importances_)[::-1]

        if self.verbose:
            print(f"✅ Selected {len(self.feature_names_)} features out of {X.shape[1]}")
            print("📊 Top Selected Features by Importance:")
            for i, name in enumerate(self.feature_names_):
                importance = self.importances_[X.columns.get_loc(name)]
                print(f"{i + 1}) \t{name} ({importance:.4f})")
        return self

    def transform(self, X):
        if self.selector is None:
            raise ValueError("You must fit the transformer before calling transform().")

        X_selected = self.selector.transform(X)
        return pd.DataFrame(X_selected, columns=self.feature_names_, index=X.index)
