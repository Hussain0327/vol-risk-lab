from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_RAW, DATE_COL, PRICE_COL


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
        self.data = pd.read_csv(
            self.filepath, parse_dates=[DATE_COL], index_col=DATE_COL
        )
        self.data.sort_index(inplace=True)
        return self.data

    def validate(self) -> bool:
        null_count = self.data[PRICE_COL].isnull().sum()
        dupe_count = self.data.index.duplicated().sum()

        if null_count > 0:
            print(f"Warning: {null_count} missing values")

        if dupe_count > 0:
            print(f"Warning: {dupe_count} duplicate dates")

        return null_count == 0 and dupe_count == 0

    def compute_returns(self, price_col: str = PRICE_COL) -> pd.Series:
        if self.data is None:
            raise ValueError("Call load() first")

        prices = self.data[price_col]
        returns = np.log(prices / prices.shift(1))
        return returns

    def get_features(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("Call load() first")

        feature_cols = [
            "VIX",
            "InterestRate",
            "ExchangeRate",
            "TEDSpread",
            "EFFR",
            "Gold",
            "Oil",
        ]
        return self.data[feature_cols]
