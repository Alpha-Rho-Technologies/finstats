import logging
import pandas as pd

# Standarize balance frequency with stats frequency:
bal_freq_standarizer = {
    'B': {'D':1,'M': 21, '2M': 42, '3M': 63, '4M':84, 'Y':252},
    'D': {'D':1,'M': 30, '2M': 60, '3M': 90, '4M':120, 'Y':360}, # If portfolio contain crypto
    'M': {'M': 1, '2M': 2, '3M': 3, '4M':4, 'Y':12},
    'W': {'M': 4, '2M': 8, '3M': 12, '4M':16, 'Y':52},
    }

# Standarize annual risk free rate with stats frequency:
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

def get_data_frequency(df: pd.Series):
    """
    Check if the DataFrame's index has a frequency set. 
    Raise an error if the frequency is not set.

    Parameters:
    df (pd.DataFrame): DataFrame to check the index frequency of.
    """
    freq = df.index.freq
    if freq is None:
        raise ValueError("The DataFrame's index does not have a frequency set. Please use df.resample() to set a frequency.")
    else:
        return freq.freqstr
    
def check_bal_freq(balance,bm_balance):
    # Check balance frequency:
    bal_freqs = []
    for bal in [balance,bm_balance]:
        balance_freq = get_data_frequency(bal)
        bal_freqs.append(balance_freq)
    bal_freq_check = all(value == bal_freqs[0] for value in bal_freqs)
    if not bal_freq_check:
        raise ValueError('Balance and Benchmark balance have a different balance frequency')
    else:
        return balance_freq

def get_stats_freq(balance,bm_balance,freq):
    balance_freq = check_bal_freq(balance=balance,bm_balance=bm_balance)
    try:
        stats_freq = bal_freq_standarizer[balance_freq][freq]
        return stats_freq
    except Exception as e:
        logging.exception(f'Stats frequency ERROR | {e}')

def get_rf(rf,stats_freq):
    
    try:
        standarized_rf = rf/rf_standarizer[stats_freq]

        return standarized_rf
    
    except Exception as e:
        logging.exception(f'ERROR Standarizing Rf | {e}')