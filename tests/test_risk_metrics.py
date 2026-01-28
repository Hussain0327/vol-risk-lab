import pytest
import numpy as np
import pandas as pd
from src.risk_metrics import RiskMetrics


class TestRiskMetrics:
    """Tests for RiskMetrics class."""

    def setup_method(self):
        """Create sample data for tests."""
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0, 0.02, 1000))

    def test_historical_var_95(self):
        """95% VaR should be approximately 5th percentile."""
        rm = RiskMetrics(self.returns)
        var_95 = rm.historical_var(0.95)

        # For normal(0, 0.02), 5th percentile ≈ -0.033
        assert -0.04 < var_95 < -0.025

    def test_cvar_less_than_var(self):
        """CVaR should be more negative than VaR (larger loss)."""
        rm = RiskMetrics(self.returns)
        var = rm.historical_var(0.95)
        cvar = rm.historical_cvar(0.95)

        assert cvar < var

    def test_var_increases_with_confidence(self):
        """99% VaR should be more extreme than 95% VaR."""
        rm = RiskMetrics(self.returns)
        var_95 = rm.historical_var(0.95)
        var_99 = rm.historical_var(0.99)

        assert var_99 < var_95

    def test_parametric_var_close_to_historical(self):
        """Parametric and historical VaR should be similar for normal data."""
        rm = RiskMetrics(self.returns)
        hist_var = rm.historical_var(0.95)
        param_var = rm.parametric_var(volatility=0.02, confidence=0.95)

        # Should be within 20% of each other
        assert abs(hist_var - param_var) / abs(hist_var) < 0.2

    def test_rolling_var_length(self):
        """Rolling VaR should return correct length."""
        rm = RiskMetrics(self.returns)
        rolling = rm.rolling_var(window=100)

        # First 99 values should be NaN
        assert rolling.isna().sum() == 99
        assert len(rolling) == len(self.returns)
