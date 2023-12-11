import logging
import pandas as pd

bal_freq_standarizer = {
    'D': {'M': 30, '2M': 60, '3M': 90, 'Y':360}, # Resampled data in daily freq and ffill
    'M': {'M': 1, '2M': 2, '3M': 3, 'Y':12},
    'W': {'M': 4, '2M': 8, '3M': 12, 'Y':52},
    }

rf_standarizer = {
  'M': 12,
  'D': 360,
  '3M': 4,
  '6M': 2,
  '4M': 3,
  '2M': 6,
  'Y': 1,
  'W': 52
}

def check_balance_frequency(df):
    # Check if the DataFrame's index is a datetime index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("The DataFrame index must be a datetime type.")

    # Calculate the time differences between consecutive dates
    time_diffs = df.index.to_series().diff().dropna()

    # Determine the most common difference
    common_diff = time_diffs.value_counts().idxmax()

    # Check if the frequency is daily or monthly
    if common_diff.days == 1:
        return 'D'
    else:
        return 'M'

def get_rf(rf,stats_freq):
    
    try:
        standarized_rf = rf/rf_standarizer[stats_freq]

        return standarized_rf
    
    except Exception as e:
        logging.exception(f'ERROR Standarizing Rf | {e}')