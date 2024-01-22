import logging
import pandas as pd
from scipy.stats import linregress

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

def default_dates(dta):
    dta = dta.sort_index()
    start_date = dta.index[0]
    end_date = dta.index[-1]
    return start_date,end_date

def check_data_index(data):
    """
    Ensures that the input data is in datetime format. Converts it to datetime if it is not.
    Raises an error if conversion is not possible.
    
    Parameters:
    data (various): The data to be checked or converted.

    Returns:
    pandas.DatetimeIndex: Data in datetime format.
    """
    if isinstance(data.index, pd.DatetimeIndex):
        return None
    else:
        try:
            data.index = pd.to_datetime(data.index)
        except ValueError:
            raise ValueError("Data cannot be converted to datetime format. Ensure data index is in datetime format.")

def format_raw_data(raw_data:pd.DataFrame):
    # Check NaN Values:
    null_val = raw_data.isna().any()
    check = len(null_val[null_val==True]) == 0
    if not check:
        logging.error('Multiple Strategy Stats | NaN Values found in asset price data')
    return raw_data.ffill().dropna(axis=1)

def get_data_frequency(df: pd.Series):
    """
    Check the Series's index frequency and infer it if not set. 
    Raise an error if the frequency cannot be inferred.

    Parameters:
    df (pd.Series): Series to check and infer the index frequency of.
    """
    freq = df.index.freq

    # If frequency is not set, try to infer it
    if freq is None:
        inferred_freq = pd.infer_freq(df.index)
        if inferred_freq is None:
            raise ValueError("Unable to infer the Series's index frequency. Please set a frequency.")
        else:
            return inferred_freq
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

def lr(x:pd.Series,y:pd.Series):
    '''linear regression model using pandas dataframes as input'''
    df = pd.concat([x,y],axis=1).dropna()
    return linregress(df.iloc[:,0],df.iloc[:,1])

def remove_outliers(data:pd.DataFrame, threshold=1.5):
    # Calculate IQR
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds for outliers
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    # Filter out the outliers
    return data[(data > lower_bound) & (data < upper_bound)]