"""
Regression model. The fast 'gatekeeper' that decides whether a setup is
obviously good, obviously bad, or borderline.

Predicts expected R-multiple `y_hat` from the feature vector, plus a `distance`
which we use as a confidence proxy. For a linear model the natural distance is
just |y_hat| relative to the decision boundary at 0; for fancier models you
would replace `predict` with one that also returns a prediction interval.

The class is small on purpose -- swap it out with gradient boosting, quantile
regression, or a small NN later by keeping the same `predict(features)` API.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.ml.feature_builder import vector_to_array


class RegressionGatekeeper:
    """Wraps a fitted regression + scaler. Use .train() to fit, .predict() to use."""

    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names
        self.model: Ridge | None = None
        self.scaler: StandardScaler | None = None

    # -- training --------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> dict:
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)
        self.model = Ridge(alpha=alpha).fit(X_scaled, y)

        train_pred = self.model.predict(X_scaled)
        rmse = float(np.sqrt(np.mean((y - train_pred) ** 2)))
        r2 = float(self.model.score(X_scaled, y))
        return {
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "train_rmse": rmse,
            "train_r2": r2,
            "coef": dict(zip(self.feature_names, self.model.coef_.tolist())),
            "intercept": float(self.model.intercept_),
        }

    # -- inference -------------------------------------------------------------

    def predict(self, features: dict[str, float]) -> dict:
        if self.model is None or self.scaler is None:
            # untrained -> emit a 'borderline' sentinel so Claude is consulted.
            return {"y_hat": 0.0, "distance": 0.0, "trained": False}

        X = vector_to_array(features, self.feature_names)
        X_scaled = self.scaler.transform(X)
        y_hat = float(self.model.predict(X_scaled)[0])
        # for ridge / linear regression we treat |y_hat| as our distance.
        distance = float(abs(y_hat))
        return {"y_hat": y_hat, "distance": distance, "trained": True}

    # -- persistence -----------------------------------------------------------

    def save(self, model_path: str, scaler_path: str) -> None:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, model_path)
        joblib.dump(self.scaler, scaler_path)

    @classmethod
    def load(cls, model_path: str, scaler_path: str) -> "RegressionGatekeeper":
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            return cls(feature_names=[])
        bundle = joblib.load(model_path)
        gk = cls(feature_names=bundle["feature_names"])
        gk.model = bundle["model"]
        gk.scaler = joblib.load(scaler_path)
        return gk
