import pandas as pd
import numpy as np
from config import TRADING_DAYS_PER_YEAR, ROLLING_WINDOWS


class FeatureEngineer:
    """Computes volatility and risk features from returns."""

    def __init__(self, returns: pd.Series):
        self.returns = returns

    def rolling_volatility(self, window: int) -> pd.Series:
        return self.returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    def realized_volatility(self, window: int = 21) -> pd.Series:
        return self.returns.shift(-window).rolling(window).std() * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )

    def lagged_features(self, n_lags: int = 5) -> pd.DataFrame:
        lags = {}

        for i in range(1, n_lags + 1):
            lags[f"return_lag_{i}"] = self.returns.shift(i)

        return pd.DataFrame(lags)

    def build_feature_matrix(self) -> pd.DataFrame:
        features = pd.DataFrame(index=self.returns.index)

        for window in ROLLING_WINDOWS:
            features[f"rolling_vol_{window}"] = self.rolling_volatility(window)

        lags = self.lagged_features()
        features = pd.concat([features, lags], axis=1)
        
        return features