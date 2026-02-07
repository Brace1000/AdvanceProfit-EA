"""Quick script to check model feature names."""
import joblib
import pandas as pd

# Load model
model = joblib.load('xgb_eurusd_h1.pkl')

# Check if it has feature names
if hasattr(model, 'feature_names_in_'):
    print('Model feature names:')
    for i, name in enumerate(model.feature_names_in_):
        print(f'{i+1}. {name}')
    print(f'\nTotal features: {len(model.feature_names_in_)}')
elif hasattr(model, 'calibrated_classifiers_'):
    # It's a calibrated model
    print('Calibrated model detected')
    base = model.calibrated_classifiers_[0].estimator
    if hasattr(base, 'feature_names_in_'):
        print('\nBase model feature names:')
        for i, name in enumerate(base.feature_names_in_):
            print(f'{i+1}. {name}')
        print(f'\nTotal features: {len(base.feature_names_in_)}')
    else:
        print('Base model has no feature_names_in_')
else:
    print('Model has no feature_names_in_')

# Load data
df = pd.read_csv('EURUSD_H1_clean.csv')
feature_cols = [c for c in df.columns if c not in ['label', 'time', 'datetime', 'timestamp']]
print(f'\n\nData file has {len(feature_cols)} features:')
for i, col in enumerate(feature_cols):
    print(f'{i+1}. {col}')
