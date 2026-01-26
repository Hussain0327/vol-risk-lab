# Volatility Risk Lab

**Core Question:** Can I predict tomorrow's volatility better than naive methods, and use that prediction to estimate portfolio risk?

## Pipeline Overview

```
Historical Prices
       ↓
   Log Returns
       ↓
Volatility Estimates ──→ Compare Methods:
       ↓                  • Rolling Window (naive baseline)
       ↓                  • EWMA (exponential weighting)
       ↓                  • GARCH (econometric)
       ↓                  • ML Models (sklearn)
       ↓
  Risk Metrics (VaR, CVaR)
       ↓
  Evaluation: Which method forecasts realized volatility best?
```

## Data

- **Source:** NASDAQ daily data (2010-2024)
- **Rows:** ~3,900 trading days
- **Columns:** Date, OHLCV, InterestRate, ExchangeRate, VIX, TEDSpread, EFFR, Gold, Oil

---

## File Implementation Guide

Files are ordered from easiest to hardest. Complete them in this sequence.

---

### 1. `config.py` (Start Here)

**Purpose:** Central place for constants and paths. No logic, just values.

**Imports:**

```python
from pathlib import Path
```

**Structure:**

```python
# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Data parameters
DATE_COL = "Date"
PRICE_COL = "Close"

# Volatility parameters
TRADING_DAYS_PER_YEAR = 252
EWMA_LAMBDA = 0.94  # Industry standard (RiskMetrics)
ROLLING_WINDOWS = [5, 21, 63]  # 1 week, 1 month, 1 quarter

# Risk parameters
VAR_CONFIDENCE_LEVELS = [0.95, 0.99]

# ML parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
```

**How to start:** Just type it out. No functions, no classes. Pure configuration.

---

### 2. `src/data_loader.py`

**Purpose:** Load CSV, parse dates, validate data, compute returns.

**Imports:**

```python
import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_RAW, DATE_COL, PRICE_COL
```

**Class Structure:**

```python
class DataLoader:
    """Loads and preprocesses NASDAQ data."""

    def __init__(self, filepath: Path = None):
        """
        Args:
            filepath: Path to CSV. Defaults to DATA_RAW / "nasdq.csv"
        """
        self.filepath = filepath or DATA_RAW / "nasdq.csv"
        self.data = None

    def load(self) -> pd.DataFrame:
        """Load CSV and parse dates."""
        # TODO: pd.read_csv with parse_dates
        # TODO: Set date as index
        # TODO: Sort by date ascending
        pass

    def validate(self) -> bool:
        """Check for missing values, duplicate dates, gaps."""
        # TODO: Check for nulls in Close column
        # TODO: Check for duplicate dates
        # TODO: Report any issues
        pass

    def compute_returns(self, price_col: str = PRICE_COL) -> pd.Series:
        """
        Compute log returns: ln(P_t / P_{t-1})

        Why log returns?
        - Additive over time (can sum daily returns for weekly)
        - Approximately normal for small returns
        - No negative prices possible
        """
        # TODO: np.log(prices / prices.shift(1))
        pass

    def get_features(self) -> pd.DataFrame:
        """Return macro features (VIX, rates, etc.) aligned with returns."""
        pass
```

**How to start:**

1. Write `load()` first. Just get the CSV into a DataFrame.
2. Print `df.head()`, `df.info()`, `df.describe()` to verify.
3. Then write `compute_returns()`.
4. Test: `returns.head()` should show NaN for first row, small decimals after.

**Key Concept:** Log returns, not simple returns. `np.log(P_t / P_{t-1})` not `(P_t - P_{t-1}) / P_{t-1}`.

---

### 3. `src/features.py`

**Purpose:** Compute volatility features (rolling, realized vol).

**Imports:**

```python
import pandas as pd
import numpy as np
from config import TRADING_DAYS_PER_YEAR, ROLLING_WINDOWS
```

**Class Structure:**

```python
class FeatureEngineer:
    """Computes volatility and risk features from returns."""

    def __init__(self, returns: pd.Series):
        """
        Args:
            returns: Log returns series with datetime index
        """
        self.returns = returns

    def rolling_volatility(self, window: int) -> pd.Series:
        """
        Compute rolling standard deviation of returns, annualized.

        Formula: std(returns, window) * sqrt(252)

        Args:
            window: Number of days (e.g., 21 for monthly)
        """
        # TODO: returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        pass

    def realized_volatility(self, window: int = 21) -> pd.Series:
        """
        Forward-looking realized vol (what we're trying to predict).
        This is the TARGET variable for ML models.

        For each day t, compute vol realized over next `window` days.
        """
        # TODO: Same formula but shifted forward
        # CAREFUL: This looks into the future. Only use as target, not feature.
        pass

    def lagged_features(self, n_lags: int = 5) -> pd.DataFrame:
        """
        Create lagged return features for ML.

        Returns DataFrame with columns: return_lag_1, return_lag_2, ...
        """
        # TODO: Loop creating shifted columns
        pass

    def build_feature_matrix(self) -> pd.DataFrame:
        """
        Combine all features into single DataFrame for modeling.

        Columns: rolling_vol_5, rolling_vol_21, rolling_vol_63,
                 return_lag_1, ..., return_lag_5, VIX, etc.
        """
        pass
```

**How to start:**

1. Write `rolling_volatility()` first with window=21.
2. Plot it against VIX from your data. They should move together.
3. If they don't correlate, something's wrong.

**Key Concept:** Annualization. Daily vol \* sqrt(252) = annual vol. Why 252? Trading days per year.

---

### 4. `src/models/ewma.py`

**Purpose:** Exponentially Weighted Moving Average volatility.

**Imports:**

```python
import pandas as pd
import numpy as np
from config import EWMA_LAMBDA
```

**Class Structure:**

```python
class EWMAVolatility:
    """
    EWMA Volatility Model (RiskMetrics approach).

    Formula: σ²_t = λ * σ²_{t-1} + (1-λ) * r²_{t-1}

    Why EWMA over rolling?
    - More weight on recent observations
    - Reacts faster to volatility changes
    - Industry standard (λ=0.94 for daily data)
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
        """
        Compute EWMA variance series.

        Initialize with sample variance of first 20 observations.
        Then iterate: var_t = lambda * var_{t-1} + (1-lambda) * r²_{t-1}
        """
        # TODO: Initialize variance
        # TODO: Loop through returns updating variance
        # TODO: Store as self.variance_series
        pass

    def get_volatility(self) -> pd.Series:
        """Return annualized volatility (sqrt of variance * sqrt(252))."""
        pass

    def forecast_next(self) -> float:
        """Forecast next period's volatility."""
        # For EWMA, forecast is just the last fitted value
        pass
```

**How to start:**

1. Implement `fit()` with a simple for loop first. Don't optimize.
2. Compare output to `returns.ewm(span=...).std()` from pandas.
3. Your manual implementation should match (approximately).

**Key Concept:** λ (lambda) controls memory. λ=0.94 means 94% weight on previous variance, 6% on new squared return.

---

### 5. `src/risk_metrics.py`

**Purpose:** Value at Risk (VaR) and Conditional VaR.

**Imports:**

```python
import pandas as pd
import numpy as np
from config import VAR_CONFIDENCE_LEVELS
```

**Class Structure:**

```python
class RiskMetrics:
    """
    Compute VaR and CVaR from returns or volatility forecasts.

    VaR: "What's the worst loss at X% confidence?"
    CVaR: "If we exceed VaR, what's the average loss?"
    """

    def __init__(self, returns: pd.Series):
        self.returns = returns

    def historical_var(self, confidence: float = 0.95) -> float:
        """
        Historical VaR: Just the percentile of actual returns.

        95% VaR = 5th percentile of returns (losses are negative)

        Args:
            confidence: 0.95 means "95% of the time, loss won't exceed this"
        """
        # TODO: np.percentile(returns, (1 - confidence) * 100)
        pass

    def historical_cvar(self, confidence: float = 0.95) -> float:
        """
        Conditional VaR (Expected Shortfall).

        Average of returns below VaR threshold.
        """
        # TODO: Get VaR, then mean of returns below VaR
        pass

    def parametric_var(self, volatility: float, confidence: float = 0.95) -> float:
        """
        Parametric VaR assuming normal distribution.

        VaR = -μ + σ * z_score

        For 95%: z ≈ 1.645
        For 99%: z ≈ 2.326
        """
        # TODO: Use scipy.stats.norm.ppf or hardcode z-scores
        pass

    def rolling_var(self, window: int = 252, confidence: float = 0.95) -> pd.Series:
        """Compute VaR on rolling window."""
        pass
```

**How to start:**

1. Write `historical_var()` first. It's one line: `np.percentile()`.
2. Verify: 95% VaR should be a negative number (it's a loss).
3. Sanity check: For NASDAQ, 95% daily VaR is probably around -2% to -3%.

**Key Concept:** VaR is a quantile. 95% VaR answers: "What's the loss I won't exceed 95% of days?"

---

### 6. `src/models/garch.py`

**Purpose:** GARCH(1,1) model using `arch` library.

**Imports:**

```python
import pandas as pd
import numpy as np
from arch import arch_model
from config import TRADING_DAYS_PER_YEAR
```

**Class Structure:**

```python
class GARCHModel:
    """
    GARCH(1,1) Volatility Model.

    σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}

    - ω (omega): Long-run variance constant
    - α (alpha): Reaction to new information (yesterday's squared return)
    - β (beta): Persistence of volatility

    α + β < 1 for stationarity. Typically α + β ≈ 0.99 for daily data.
    """

    def __init__(self, p: int = 1, q: int = 1):
        """
        Args:
            p: GARCH lag order (usually 1)
            q: ARCH lag order (usually 1)
        """
        self.p = p
        self.q = q
        self.model = None
        self.fitted = None

    def fit(self, returns: pd.Series) -> "GARCHModel":
        """
        Fit GARCH model to returns.

        Note: arch library expects returns in percentage (multiply by 100)
        or set rescale=True.
        """
        # TODO: Create arch_model with vol='Garch', p=self.p, q=self.q
        # TODO: Fit with disp='off' to suppress output
        # TODO: Store fitted model
        pass

    def get_params(self) -> dict:
        """Return fitted parameters (omega, alpha, beta)."""
        pass

    def get_conditional_volatility(self) -> pd.Series:
        """Return fitted conditional volatility series."""
        # TODO: self.fitted.conditional_volatility
        pass

    def forecast(self, horizon: int = 1) -> pd.DataFrame:
        """Forecast volatility for next `horizon` periods."""
        # TODO: self.fitted.forecast(horizon=horizon)
        pass

    def summary(self) -> None:
        """Print model summary."""
        pass
```

**How to start:**

1. Read the `arch` library documentation first: https://arch.readthedocs.io/
2. Start with the simplest example from docs, get it working.
3. Then wrap it in your class.

**Key Concept:** GARCH captures "volatility clustering" - high vol days follow high vol days.

---

### 7. `src/models/ml_vol.py`

**Purpose:** ML-based volatility forecasting.

**Imports:**

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from config import RANDOM_STATE, TEST_SIZE
```

**Class Structure:**

```python
class MLVolatilityModel:
    """
    ML-based volatility forecasting.

    Target: Next-day (or next-N-day) realized volatility
    Features: Lagged returns, lagged vol, VIX, macro variables
    """

    def __init__(self, model_type: str = "ridge"):
        """
        Args:
            model_type: "ridge", "rf" (random forest), or "gbm"
        """
        self.model_type = model_type
        self.model = self._get_model()
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _get_model(self):
        """Return sklearn model based on model_type."""
        models = {
            "ridge": Ridge(alpha=1.0),
            "rf": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=RANDOM_STATE
            ),
            "gbm": GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=RANDOM_STATE
            )
        }
        return models.get(self.model_type)

    def prepare_data(self, features: pd.DataFrame, target: pd.Series):
        """
        Align features and target, drop NaNs, split train/test.

        IMPORTANT: No shuffling. Time series must respect order.
        """
        # TODO: Align on index
        # TODO: Drop rows with any NaN
        # TODO: Split by time, not random
        pass

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "MLVolatilityModel":
        """
        Fit model with scaling.
        """
        # TODO: Fit scaler on X_train
        # TODO: Transform X_train
        # TODO: Fit model
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict volatility."""
        # TODO: Scale X, then predict
        pass

    def get_feature_importance(self) -> pd.Series:
        """Return feature importances (for tree models)."""
        pass
```

**How to start:**

1. Get Ridge working first. It's the simplest.
2. Create a tiny feature matrix manually (3 columns), verify fit/predict work.
3. Then add RandomForest.
4. Then integrate with FeatureEngineer class.

**Key Concept:** Time series split. Never shuffle. Train on past, test on future.

---

### 8. `src/evaluation.py`

**Purpose:** Compare model performance.

**Imports:**

```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
```

**Class Structure:**

```python
class ModelEvaluator:
    """Evaluate and compare volatility forecasts."""

    def __init__(self, actual: pd.Series):
        """
        Args:
            actual: Realized volatility (ground truth)
        """
        self.actual = actual
        self.forecasts = {}  # Store forecasts from different models

    def add_forecast(self, name: str, forecast: pd.Series) -> None:
        """Add a model's forecast for comparison."""
        self.forecasts[name] = forecast

    def compute_metrics(self, forecast: pd.Series) -> dict:
        """
        Compute error metrics.

        Returns:
            dict with MAE, RMSE, MAPE
        """
        # Align series
        aligned = pd.concat([self.actual, forecast], axis=1).dropna()
        actual = aligned.iloc[:, 0]
        pred = aligned.iloc[:, 1]

        return {
            "MAE": mean_absolute_error(actual, pred),
            "RMSE": np.sqrt(mean_squared_error(actual, pred)),
            "MAPE": np.mean(np.abs((actual - pred) / actual)) * 100
        }

    def compare_all(self) -> pd.DataFrame:
        """Return comparison table of all models."""
        # TODO: Loop through self.forecasts, compute metrics for each
        pass

    def var_backtest(self, returns: pd.Series, var_series: pd.Series,
                     confidence: float = 0.95) -> dict:
        """
        Backtest VaR predictions.

        Count how often actual returns breached VaR.
        Expected breach rate = 1 - confidence (e.g., 5% for 95% VaR)
        """
        # TODO: Count days where return < VaR
        # TODO: Compare to expected breach rate
        pass
```

**How to start:**

1. Write `compute_metrics()` first with just MAE.
2. Manually create two fake series, verify the math.
3. Then add RMSE, MAPE.

**Key Concept:** Lower MAE/RMSE = better. VaR backtest: if 95% VaR is breached way more than 5% of days, your model underestimates risk.

---

### 9. `src/backtester.py`

**Purpose:** Walk-forward backtesting framework.

**Imports:**

```python
import pandas as pd
import numpy as np
from typing import Callable
```

**Class Structure:**

```python
class WalkForwardBacktester:
    """
    Walk-forward backtesting for time series models.

    Why walk-forward?
    - Simulates real trading: only use data available at time t
    - Avoids look-ahead bias
    - More realistic than single train/test split
    """

    def __init__(self,
                 min_train_size: int = 252,  # 1 year minimum
                 step_size: int = 21,         # Refit monthly
                 expanding: bool = True):     # Expanding vs sliding window
        """
        Args:
            min_train_size: Minimum training window
            step_size: Days between refits
            expanding: If True, training window grows. If False, slides.
        """
        self.min_train_size = min_train_size
        self.step_size = step_size
        self.expanding = expanding

    def generate_splits(self, data: pd.DataFrame):
        """
        Generate train/test indices for walk-forward.

        Yields:
            Tuples of (train_idx, test_idx)
        """
        # TODO: Loop from min_train_size to end
        # TODO: Yield expanding or sliding windows
        pass

    def run(self,
            data: pd.DataFrame,
            model_fn: Callable,  # Function that fits and returns predictions
            target_col: str) -> pd.Series:
        """
        Run walk-forward backtest.

        Args:
            data: Full dataset
            model_fn: Function(train_data) -> predictions for next period
            target_col: Column name of target variable

        Returns:
            Series of out-of-sample predictions
        """
        pass
```

**How to start:**

1. Write `generate_splits()` first. Just print the indices.
2. Verify: First split should be [0:252] train, [252:273] test (approximately).
3. Second split: [0:273] train (expanding) or [21:273] train (sliding).

**Key Concept:** Never train on future data. At time t, you only know data up to t-1.

---

### 10. `tests/test_risk_metrics.py`

**Purpose:** Unit tests for risk calculations.

**Imports:**

```python
import pytest
import numpy as np
import pandas as pd
from src.risk_metrics import RiskMetrics
```

**Structure:**

```python
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

        assert cvar < var  # Both negative, CVaR more negative

    def test_var_increases_with_confidence(self):
        """99% VaR should be more extreme than 95% VaR."""
        rm = RiskMetrics(self.returns)
        var_95 = rm.historical_var(0.95)
        var_99 = rm.historical_var(0.99)

        assert var_99 < var_95  # Both negative, 99% more negative
```

**How to start:**

1. Run `pytest tests/` to see what happens (should fail initially).
2. Write one test, make it pass, repeat.

---

## Notebooks

### `notebooks/01_eda.ipynb`

Use this for exploration and visualization. Suggested structure:

```python
# Cell 1: Imports and data loading
from src.data_loader import DataLoader

loader = DataLoader()
df = loader.load()
returns = loader.compute_returns()

# Cell 2: Basic stats
print(returns.describe())
print(f"Skewness: {returns.skew():.3f}")
print(f"Kurtosis: {returns.kurtosis():.3f}")  # Should be > 3 (fat tails)

# Cell 3: Visualize returns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
returns.plot(ax=axes[0], title="Daily Returns")
returns.hist(bins=100, ax=axes[1], title="Return Distribution")

# Cell 4: Volatility clustering
# Plot rolling vol, EWMA vol, and VIX on same chart

# Cell 5: Correlation analysis
# Heatmap of features vs realized vol
```

---

## Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
arch>=6.0
pytest>=7.0
jupyter>=1.0
```

---

## Execution Order

1. `config.py` (10 min)
2. `data_loader.py` (1-2 hours)
3. `01_eda.ipynb` - explore your data (2-3 hours)
4. `features.py` (2-3 hours)
5. `ewma.py` (1-2 hours)
6. `risk_metrics.py` (2-3 hours)
7. `test_risk_metrics.py` (1 hour)
8. `garch.py` (2-3 hours)
9. `ml_vol.py` (3-4 hours)
10. `evaluation.py` (2-3 hours)
11. `backtester.py` (3-4 hours)
12. `main.py` - tie it all together

**Total estimated time: 25-35 hours over 5-6 weeks**
