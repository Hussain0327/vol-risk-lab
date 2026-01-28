import pandas as pd

# import numpy as np
from typing import Callable, Generator, Tuple


class WalkForwardBacktester:
    """
    Walk-forward backtesting for time series models.
    """

    def __init__(
        self, min_train_size: int = 252, step_size: int = 21, expanding: bool = True
    ):
        self.min_train_size = min_train_size
        self.step_size = step_size
        self.expanding = expanding

    def generate_splits(
        self, data: pd.DataFrame
    ) -> Generator[Tuple[range, range], None, None]:
        n = len(data)

        train_start = 0
        train_end = self.min_train_size

        while train_end + self.step_size <= n:
            test_end = min(train_end + self.step_size, n)

            train_idx = range(train_start, train_end)
            test_idx = range(train_end, test_end)

            yield train_idx, test_idx

            # Move forward
            train_end += self.step_size

            # Sliding window: move train_start forward too
            if not self.expanding:
                train_start += self.step_size

    def run(self, data: pd.DataFrame, model_fn: Callable, target_col: str) -> pd.Series:
        predictions = []
        indices = []

        for train_idx, test_idx in self.generate_splits(data):
            train_data = data.iloc[list(train_idx)]
            test_data = data.iloc[list(test_idx)]

            # Get predictions from model
            preds = model_fn(train_data)

            # Store predictions with their indices
            predictions.extend(preds)
            indices.extend(test_data.index)

        return pd.Series(predictions, index=indices)
