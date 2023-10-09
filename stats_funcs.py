import calendar
from common_functions import lr,np,pd
import datetime as dt
import logging

def cal_geo_mean(balance:pd.Series):
    try:
        geo_mean = np.exp(np.log(1+balance.pct_change()).mean()) - 1
        return geo_mean
    
    except Exception as e:
        logging.exception(f'ERROR Calc mean | {e}')

def calc_corr(balance:pd.Series,bm_balance:pd.Series):
    try:
        corr = balance.pct_change().corr(bm_balance.pct_change())
        return corr
    
    except Exception as e:
        logging.exception(f'ERROR Calc correlation | {e}')

def cal_geo_std(balance:pd.Series):
    try:
        geo_std = np.exp(np.log(1+balance.pct_change()).std()) - 1
        return geo_std
    
    except Exception as e:
        logging.exception(f'ERROR Calc Geo STD | {e}')

def calc_downside_dev(balance:pd.Series):
    try:
        log_pct = np.log(1+balance.pct_change())
        mean = log_pct.mean()
        dowside_dev = np.exp(log_pct[log_pct<mean].std())-1
        return dowside_dev
    
    except Exception as e:
        logging.exception(f'ERROR Calc Downside STD | {e}')

def calc_pos_returns_pct(balance:pd.Series):
    try:
        bal_pct = balance.pct_change()
        pos = bal_pct[bal_pct>0].count()/bal_pct.count()
        return pos
    
    except Exception as e:
        logging.exception(f'ERROR Calc Positive % returns | {e}')

def calc_es(balance:pd.Series,level = 99):
    try:
        percentail = 1 - level/100
        pct = balance.pct_change()
        es_99 = pct.quantile(percentail)
        return es_99
    
    except Exception as e:
        logging.exception(f'ERROR Calc ES99 | {e}')

def calc_max_return(balance:pd.Series):
    try:
        pct = balance.pct_change()
        max = pct.max()
        return max
    
    except Exception as e:
        logging.exception(f'ERROR Calc Max Return | {e}')

def calc_min_return(balance:pd.Series):
    try:
        pct = balance.pct_change()
        min = pct.min()
        return min
    
    except Exception as e:
        logging.exception(f'ERROR Calc Min Return  | {e}')

def get_rf(rf,freq):
    
    try:
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

        standarized_rf = rf/rf_standarizer[freq]

        return standarized_rf
    
    except Exception as e:
        logging.exception(f'ERROR Standarizing Rf | {e}')

def calc_max_dd(balance:pd.Series) -> float:
    try:
        # Calculate Maximum Drawdown
        running_max = balance[0]
        max_drawdown = 0
        for x in balance:
            if x > running_max:
                running_max = x

            drawdown = (x - running_max) / running_max
            
            if drawdown < max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    except Exception as e:
        logging.exception(f'ERROR Calc Max DD | {e}')

def calc_losing_streak(balance:pd.Series) -> int:
    balance_pct = balance.pct_change()
    try:
        # Calculate Maximum Losing Streak
        current_streak = 0
        max_losing_streak = 0

        for x in balance_pct:
            if x < 0:
                current_streak += 1
                max_losing_streak = max(max_losing_streak, current_streak)
            else:
                current_streak = 0
        
        return max_losing_streak
    except Exception as e:
        logging.exception(f'ERROR Calc losing streak | {e}')

def calc_recovery(balance:pd.Series) -> int:
    """
    Calculate the recovery period from the maximum drawdown in a given array.

    Parameters:
    - arr: List of equity values.

    Returns:
    - recovery_period: Number of steps to recover from the maximum drawdown. 
                      Returns None if it never recovers.
    """
    try:
        peak = balance[0]
        max_drawdown = 0
        max_drawdown_peak = 0

        # Identify the maximum drawdown
        for x in balance:
            if x > peak:
                peak = x

            current_drawdown = (peak - x) / peak
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
                max_drawdown_peak = peak

        # Calculate the recovery period
        recovery_period = None
        if max_drawdown > 0:  # Check to prevent division by zero
            drawdown_started = False
            steps_after_trough = 0

            for x in balance:
                if x == max_drawdown_peak and not drawdown_started:
                    drawdown_started = True

                elif drawdown_started and x >= max_drawdown_peak:
                    recovery_period = steps_after_trough
                    break

                elif drawdown_started:
                    steps_after_trough += 1

        return recovery_period
    except Exception as e:
        logging.exception(f'ERROR Calc Recovery Max DD | {e}')

def calc_info_ratio(balance:pd.Series,bm_balance:pd.Series,log_returns = True) -> float:
    try:
        if log_returns:
            balance_pct = np.log(1+balance.pct_change())
            bm_balance_pct = np.log(1+bm_balance.pct_change())
            outperf_bm = balance_pct - bm_balance_pct
            outperf_mean = np.exp(outperf_bm.mean())-1
            outperf_std = np.exp(outperf_bm.std())-1
        
        else:
            balance_pct = balance.pct_change()
            bm_balance_pct = bm_balance.pct_change()
            outperf_bm = balance_pct - bm_balance_pct
            outperf_mean = outperf_bm.mean()
            outperf_std = outperf_bm.std()

        if outperf_std > 0:
            info_ratio = outperf_mean/outperf_std
            return info_ratio
        else:
            return np.nan
        
    except Exception as e:
        logging.exception(f'ERROR Calc Info Ratio | {e}')

def calc_beta_alpha(balance,bm_balance,log_returns=True):

    try:
        if log_returns:
            balance_pct = np.log(1+balance.pct_change())
            bm_balance_pct = np.log(1+bm_balance.pct_change())
            linreg = lr(bm_balance_pct,balance_pct)
            beta = linreg[0]
            alpha = np.exp(linreg[1])-1
        
        else:
            balance_pct = balance.pct_change()
            bm_balance_pct = bm_balance.pct_change()
            linreg = lr(bm_balance_pct,balance_pct)
            beta = linreg[0]
            alpha = linreg[1]
        
        return {'alpha':alpha,'beta':beta}
        
    except Exception as e:
        logging.exception(f'ERROR Calc Info Ratio | {e}')

def calc_jensen_alpha(balance:pd.Series,bm_balance:pd.Series,rf:float, beta:float) -> float:
    try:
        bm_geo_return = cal_geo_mean(balance=bm_balance)
        geo_return = cal_geo_mean(balance=balance)
        bm_excess_return = bm_geo_return - rf
        
        jensen_alpha = geo_return - (rf + beta*bm_excess_return)
        return jensen_alpha
    except Exception as e:
        logging.exception(f'ERROR Calc Jensen Alpha | {e}')