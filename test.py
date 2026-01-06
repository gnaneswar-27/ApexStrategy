import pandas as pd

df = pd.read_parquet(
    "data/processed/race_driver_level/race_driver_features.parquet"
)

df.head()
df.shape
