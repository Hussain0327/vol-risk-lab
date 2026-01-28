from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESED = PROJECT_ROOT / "data" / "processed"

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
