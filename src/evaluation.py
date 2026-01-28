import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ModelEvaluator:
    """Evaluate and compare volatility forecasts."""

    def __init__(self, actual: pd.Series):
        self.actual = actual
        self.forecasts = {}

    def add_forecast(self, name: str, forecast: pd.Series) -> None:
        self.forecasts[name] = forecast

    def compute_metrics(self, forecast: pd.Series) -> dict:
        aligned = pd.concat([self.actual, forecast], axis=1).dropna()
        actual = aligned.iloc[:, 0]
        pred = aligned.iloc[:, 1]

        return {
            "MAE": mean_absolute_error(actual, pred),
            "RMSE": np.sqrt(mean_squared_error(actual, pred)),
            "MAPE": np.mean(np.abs((actual - pred) / actual)) * 100,
        }

    def compare_all(self) -> pd.DataFrame:
        results = {}
        for name, forecast in self.forecasts.items():
            results[name] = self.compute_metrics(forecast)
        return pd.DataFrame(results).T

    def var_backtest(
        self, returns: pd.Series, var_series: pd.Series, confidence: float = 0.95
    ) -> dict:
        # Align returns and VaR
        aligned = pd.concat([returns, var_series], axis=1).dropna()
        actual_returns = aligned.iloc[:, 0]
        var_predictions = aligned.iloc[:, 1]

        # Count breaches (actual loss worse than VaR)
        breaches = (actual_returns < var_predictions).sum()
        total_days = len(aligned)
        breach_rate = breaches / total_days
        expected_rate = 1 - confidence

        return {
            "breaches": breaches,
            "total_days": total_days,
            "breach_rate": breach_rate,
            "expected_rate": expected_rate,
            "ratio": breach_rate / expected_rate,  # Should be close to 1.0
        }
