#!/usr/bin/env python3.9
"""LogisticRegression Experiment"""

import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(f"Python version: {sys.version}")  # target is 3.9.7
print()
print(f"pandas version: {pd.__version__}")  # target is 1.3.4
print(f"numpy version: {np.__version__}")  # target is 1.19.5
print(f"scikit-learn version: {sklearn.__version__}")  # target is 1.0.0
print()
print(f"Solver being used: '{LogisticRegression().solver}'")

diabetes_df = pd.read_csv("pima_indians_diabetes.csv")
# diabetes_df = diabetes_df.drop(["pregnant"], axis=1)

y = diabetes_df["test"]
X = diabetes_df.drop("test", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = StandardScaler()
lr = LogisticRegression()

rfe = RFE(estimator=LogisticRegression(max_iter=500), n_features_to_select=3, verbose=1)

rfe.fit(X_train, y_train)

print(dict(zip(X.columns, rfe.ranking_)))

print(X.columns[rfe.support_])

acc = accuracy_score(y_test, rfe.predict(X_test))
print(f"{acc:.1%} accuracy on test set.")
