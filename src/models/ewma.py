import numpy as np
import pandas as pd

from config import EWMA_LAMBDA


class EWMAVolatility:
    """
    Formula: o²_t = λ * o²_{t-1} + (1-λ) * r²_{t-1}
    """

    def __init__(self, lambda_param: float = EWMA_LAMBDA):
        """
        Args:
            lambda_param: Decay factor. Higher = more weight on past.
            0.94 is RiskMetrics standard for daily data.
        """
        self.lambda_param = lambda_param
        self.variance_series = None

    def fit(self, returns: pd.Series) -> "EWMAVolatility":
        variance = returns.iloc[:20].var()
        variances = [variance]

        for i in range(1, len(returns)):
            variance = (
                self.lambda_param * variance
                + (1 - self.lambda_param) * returns.iloc[i - 1] ** 2
            )
            variances.append(variance)
        self.variance_series = pd.Series(variances, index=returns.index)
        return self

    def get_volatility(self) -> pd.Series:
        return np.sqrt(self.variance_series) * np.sqrt(252)

    def forecast_next(self) -> float:
        return np.sqrt(self.variance_series.iloc[-1]) * np.sqrt(252)
