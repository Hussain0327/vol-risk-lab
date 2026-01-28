import pandas as pd
import numpy as np
from arch import arch_model
from config import TRADING_DAYS_PER_YEAR


class GARCHModel:
    """
    GARCH(1,1) Volatility Model.
    """

    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
        self.model = None
        self.fitted = None

    def fit(self, returns: pd.Series) -> "GARCHModel":
        # arch expects returns * 100 (percentage)
        self.model = arch_model(returns * 100, vol="Garch", p=self.p, q=self.q)
        self.fitted = self.model.fit(disp="off")
        return self

    def get_params(self) -> dict:
        params = self.fitted.params
        return {
            "omega": params["omega"],
            "alpha": params["alpha[1]"],
            "beta": params["beta[1]"],
        }

    def get_conditional_volatility(self) -> pd.Series:
        # Divide by 100 to convert back from percentage, then annualize
        return self.fitted.conditional_volatility / 100 * np.sqrt(TRADING_DAYS_PER_YEAR)

    def forecast(self, horizon: int = 1) -> pd.DataFrame:
        return self.fitted.forecast(horizon=horizon)

    def summary(self) -> None:
        print(self.fitted.summary())
