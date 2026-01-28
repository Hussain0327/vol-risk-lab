import numpy as np
import pandas as pd
from scipy.stats import norm

# from config import VAR_CONFIDENCE_LEVELS


class RiskMetrics:
    """Compute VaR and CVaR from returns or volatility forecasts."""

    def __init__(self, returns: pd.Series):
        self.returns = returns

    def historical_var(self, confidence: float = 0.95) -> float:
        percentile = (1 - confidence) * 100
        return np.percentile(self.returns, percentile)

    def historical_cvar(self, confidence: float = 0.95) -> float:
        var = self.historical_var(confidence)
        return self.returns[self.returns <= var].mean()

    def parametric_var(self, volatility: float, confidence: float = 0.95) -> float:
        """
        Parametric VaR assuming normal distribution.
        """

        z_score = norm.ppf(1 - confidence)  # -1.645 for 95%
        mean = self.returns.mean()
        return mean + z_score * volatility  # Will be negative

    def rolling_var(self, window: int = 252, confidence: float = 0.95) -> pd.Series:
        """Compute VaR on rolling window."""
        percentile = (1 - confidence) * 100
        return self.returns.rolling(window).apply(
            lambda x: np.percentile(x, percentile)
        )
