import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# from sklearn.model_selection import TimeSeriesSplit
from config import RANDOM_STATE, TEST_SIZE


class MLVolatilityModel:
    """
    Target: Next-day (or next-N-day) realized volatility
    Features: Lagged returns, lagged vol, VIX, macro variables
    """

    def __init__(self, model_type: str = "ridge"):
        self.model_type = model_type
        self.model = self._get_model()
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _get_model(self):
        models = {
            "ridge": Ridge(alpha=1.0),
            "rf": RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=RANDOM_STATE
            ),
            "gbm": GradientBoostingRegressor(
                n_estimators=100, max_depth=5, random_state=RANDOM_STATE
            ),
        }
        return models.get(self.model_type)

    def prepare_data(self, features: pd.DataFrame, target: pd.Series):
        data = pd.concat([features, target.rename("target")], axis=1).dropna()

        X = data.drop("target", axis=1)
        y = data["target"]

        split_idx = int(len(data) * (1 - TEST_SIZE))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        return X_train, X_test, y_train, y_test

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "MLVolatilityModel":
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> pd.Series:
        if self.model_type == "ridge":
            return pd.Series(self.model.coef_)
        return pd.Series(self.model.feature_importances_)
