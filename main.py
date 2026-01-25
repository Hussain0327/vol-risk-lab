from pathlib import Path

import pandas as pd

df = Path("data/raw/nasdq.csv")

df = pd.read_csv(df)
