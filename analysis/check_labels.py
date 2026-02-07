"""Check label distribution across entire dataset."""
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import get_config

config = get_config()

# Load data
data_file = Path(config.get("paths.data_processed_file", "EURUSD_H1_clean.csv"))
df = pd.read_csv(data_file)

print("=" * 80)
print("LABEL DISTRIBUTION ANALYSIS")
print("=" * 80)

# Overall distribution
print(f"\nTotal samples: {len(df)}")
print(f"\nOverall label distribution:")
for label in sorted(df['label'].unique()):
    count = (df['label'] == label).sum()
    pct = count / len(df)
    print(f"  Class {label}: {count} ({pct:.1%})")

# Split points
train_ratio = float(config.get("training.train_ratio", 0.6))
val_ratio = float(config.get("training.val_ratio", 0.2))

n = len(df)
train_end = int(n * train_ratio)
val_end = int(n * (train_ratio + val_ratio))

print(f"\n" + "=" * 80)
print("LABEL DISTRIBUTION BY SPLIT (Chronological)")
print("=" * 80)

splits = {
    'Train': (0, train_end),
    'Validation': (train_end, val_end),
    'Holdout': (val_end, n)
}

for split_name, (start, end) in splits.items():
    df_split = df.iloc[start:end]
    print(f"\n{split_name} ({start}:{end}, n={end-start}):")

    for label in sorted(df['label'].unique()):
        count = (df_split['label'] == label).sum()
        pct = count / len(df_split) if len(df_split) > 0 else 0
        print(f"  Class {label}: {count} ({pct:.1%})")

# Check temporal distribution (by year or quarter)
if 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year

    print(f"\n" + "=" * 80)
    print("LABEL DISTRIBUTION BY YEAR")
    print("=" * 80)

    for year in sorted(df['year'].unique()):
        df_year = df[df['year'] == year]
        print(f"\nYear {year} (n={len(df_year)}):")
        for label in sorted(df['label'].unique()):
            count = (df_year['label'] == label).sum()
            pct = count / len(df_year) if len(df_year) > 0 else 0
            print(f"  Class {label}: {count} ({pct:.1%})")

print("\n" + "=" * 80)
