from art_utils.common_functions import lr,np,pd
from fin_stats.src.utils import *
import datetime as dt
import logging

class fin_stats:

    def __init__(self,balance:pd.Series,stats_freq:str,bm_balance:pd.Series=None) -> None:
        """
        Initialize the FinStats object with financial data and calculate relevant statistics.

        Parameters:
        balance (pd.Series): A Pandas Series representing the balance of an investment over time.
        stats_freq (str): Frequency string to define the periodicity for calculating returns.
        bm_balance (pd.Series, optional): A Pandas Series representing the benchmark balance over time. Defaults to None.

        The constructor tries to calculate various statistics such as returns and logarithmic returns for both 
        the balance and the benchmark balance (if provided). If any error occurs during these calculations, 
        it logs the exception with a descriptive message.
        """
        try:
            stats_freq = get_stats_freq(balance=balance,
                                        bm_balance=bm_balance,
                                        freq=stats_freq)
            
            self.balance = balance
            self.returns = balance.pct_change(stats_freq, fill_method=None)
            self.log_returns = np.log(1 + self.returns)
            
            if bm_balance is not None:
                self.bm_balance = bm_balance
                self.bm_returns = bm_balance.pct_change(stats_freq, fill_method=None)
                self.bm_log_returns = np.log(1 + self.bm_returns)
        
        except Exception as e:
            logging.exception(f'Failed to Initialize fin_stats | {e}')
        
    def mean_returns(self,geometric = True):
        try:
            if geometric:
                mean = np.exp(self.log_returns.mean()) - 1
            else:
                mean = self.returns.mean()
            return mean
        
        except Exception as e:
            logging.exception(f'ERROR calculating mean returns | {e}')

    def correlation(self):
        try:
            corr = self.returns.corr(self.bm_returns)
            return corr
        
        except Exception as e:
            logging.exception(f'ERROR calculating correlation | {e}')

    def returns_standard_deviation(self,geometric = True):
        try:
            if geometric:
                std = np.exp(np.log(1+self.log_returns).std()) - 1
            else:
                std = self.returns.std()
            return std
        
        except Exception as e:
            logging.exception(f'ERROR calculating returns standard deviation | {e}')

    def downside_deviation(self, geometric = True):
        try:
            if geometric:
                mean = self.log_returns.mean()
                dowside_dev = np.exp(self.log_returns[self.log_returns<mean].std())-1
            else:
                mean = self.mean_returns(geometric=False)
                dowside_dev = self.returns[self.returns<mean].std()
            
            return dowside_dev
        
        except Exception as e:
            logging.exception(f'ERROR calculating Downside standard deviation | {e}')

    def positive_returns_pct(self):
        try:
            pos = self.returns[self.returns>0].count()/self.returns.count()
            return pos
        
        except Exception as e:
            logging.exception(f'ERROR calculating Positive % returns | {e}')

    def es(self,level = 99):
        try:
            percentail = 1 - level/100
            es_99 = self.returns.quantile(percentail)
            return es_99
        
        except Exception as e:
            logging.exception(f'ERROR calculating ES99 | {e}')

    def max_return(self):
        try:
            max = self.returns.max()
            return max
        
        except Exception as e:
            logging.exception(f'ERROR calculating Max Return | {e}')

    def min_return(self):
        try:
            min = self.returns.min()
            return min
        
        except Exception as e:
            logging.exception(f'ERROR calculating Min Return  | {e}')

    def max_dd(self) -> float:
        """
        Calculates the maximum drawdown of a given balance series.

        Parameters:
        balance (pd.Series): A pandas series containing the balance values.

        Returns:
        float: The maximum drawdown as a percentage.
        """
        try:
            # Calculate Maximum Drawdown
            running_max = self.balance.cummax()
            drawdown = (self.balance - running_max) / running_max
            max_drawdown = drawdown.min()

            return max_drawdown

        except Exception as e:
            logging.exception(f'ERROR calculating Max DD | {e}')

    def losing_streak(self) -> int:
        balance_pct = self.returns
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
            logging.exception(f'ERROR calculating losing streak | {e}')

    def recovery(self) -> int:
        """
        Calculate the recovery period from the maximum drawdown in a given array.

        Returns:
        - recovery_period: Number of steps to recover from the maximum drawdown. 
                           Returns None if it never recovers.
        """
        try:
            peak = self.balance.iloc[0]
            max_drawdown = 0
            drawdown_start_index = 0

            # Identify the maximum drawdown
            for index, x in enumerate(self.balance):
                if x > peak:
                    peak = x
                current_drawdown = (peak - x) / peak
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown
                    drawdown_start_index = index

            # Calculate the recovery period
            if max_drawdown == 0:
                return None  # No drawdown occurred

            for recovery_index, x in enumerate(self.balance[drawdown_start_index:], start=drawdown_start_index):
                if x >= peak:
                    return recovery_index - drawdown_start_index

            return None  # Never recovers
        except Exception as e:
            logging.exception(f'ERROR calculating Recovery Max DD | {e}')

    def info_ratio(self,log_returns = True) -> float:
        try:
            if log_returns:
                outperf_bm = self.log_returns - self.bm_log_returns
                outperf_mean = np.exp(outperf_bm.mean())-1
                outperf_std = np.exp(outperf_bm.std())-1
            
            else:
                outperf_bm = self.returns - self.bm_returns
                outperf_mean = outperf_bm.mean()
                outperf_std = outperf_bm.std()

            if outperf_std > 0:
                info_ratio = outperf_mean/outperf_std
                return info_ratio
            else:
                return np.nan
            
        except Exception as e:
            logging.exception(f'ERROR calculating Info Ratio | {e}')

    def beta_alpha(self,geometric=True):

        try:
            if geometric:
                linreg = lr(self.bm_log_returns,self.log_returns)
                beta = linreg[0]
                alpha = np.exp(linreg[1])-1
            
            else:
                linreg = lr(self.bm_returns,self.returns)
                beta = linreg[0]
                alpha = linreg[1]
            
            return {'alpha':alpha,'beta':beta}
            
        except Exception as e:
            logging.exception(f'ERROR calculating Info Ratio | {e}')

    def jensen_alpha(self, rf:float, geometric = True) -> float:
        try:
            if geometric:
                beta = self.beta_alpha(geometric=True)['beta']
                bm_mean_returns = np.exp(np.log(1+self.bm_log_returns).mean())-1
                mean_returns = self.mean_returns(geometric=True)
            else:
                beta = self.beta_alpha(geometric=False)['beta']
                bm_mean_returns = self.bm_returns.mean()
                mean_returns = self.mean_returns(geometric=False)
                
            bm_excess_return = bm_mean_returns - rf
            jensen_alpha = mean_returns - (rf + beta*bm_excess_return)
            return jensen_alpha
        
        except Exception as e:
            logging.exception(f'ERROR calculating Jensen Alpha | {e}')