"""Small wrapper for training and saving a scikit-learn LinearRegression model.

Provides:
- train_linear_regression(X_train, y_train): fits and returns (model, scaler)
- save_model(path, model, scaler): saves model and scaler together using joblib
"""
from typing import Tuple
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def train_linear_regression(X, y) -> Tuple[LinearRegression, StandardScaler]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(Xs, y)
    return model, scaler


def save_model(path: str, model, scaler):
    joblib.dump({'model': model, 'scaler': scaler}, path)
