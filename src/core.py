from finstats.src.utils import *
import numpy as np
import logging

class fin_stats:

    def __init__(self,balance:pd.Series,stats_freq:str,rf:float,bm_balance:pd.Series=None) -> None:
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
            
            self.rf = rf
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
            return None

    def correlation(self):
        try:
            corr = self.returns.corr(self.bm_returns)
            return corr
        
        except Exception as e:
            logging.exception(f'ERROR calculating correlation | {e}')
            return None

    def returns_standard_deviation(self,geometric = True):
        try:
            if geometric:
                std = np.exp(self.log_returns.std()) - 1
            else:
                std = self.returns.std()
            return std
        
        except Exception as e:
            logging.exception(f'ERROR calculating returns standard deviation | {e}')
            return None
    
    def sharpe_ratio(self,geometric = True):
        try:
            if geometric:
                mean = self.mean_returns()
                std = self.returns_standard_deviation()
                excess_return = mean-self.rf
                sharpe_ratio = excess_return/std
                
            else:
                mean = self.returns.mean()
                std = self.returns.std()
                excess_return = mean-self.rf
                sharpe_ratio = excess_return/std
            return sharpe_ratio
        
        except Exception as e:
            logging.exception(f'ERROR calculating Sharpe Ratio | {e}')
            return None

    def sortino_ratio(self,geometric = True):
        try:
            if geometric:
                mean = self.mean_returns()
                downside_std = self.downside_deviation()
                excess_return = mean-self.rf
                sortino_ratio = excess_return/downside_std
                
            else:
                mean = self.returns.mean()
                downside_std = self.downside_deviation(geometric=False)
                excess_return = mean - self.rf
                sortino_ratio = excess_return/downside_std
            
            return sortino_ratio
        
        except Exception as e:
            logging.exception(f'ERROR calculating Sortino Ratio | {e}')
            return None
        
    def calmar_ratio(self,geometric = True):
        try:
            downside = self.min_return()
            if downside < 0:
                if geometric:
                    mean = self.mean_returns()
                    
                    excess_return = mean-self.rf
                    calmar_ratio = excess_return/downside
                    
                else:
                    mean = self.returns.mean()
                    downside = self.min_return()
                    excess_return = mean-self.rf
                    calmar_ratio = excess_return/downside
                return abs(calmar_ratio)
            else:
                return np.nan
        
        except Exception as e:
            logging.exception(f'ERROR calculating Calmar Ratio | {e}')
            return None

    def downside_deviation(self, geometric = True):
        try:
            if geometric:
                mean = self.log_returns.mean()
                excess_log_returns = self.log_returns - mean
                downside_log_returns = excess_log_returns[excess_log_returns < 0]
                downside_dev = np.exp(np.sqrt((downside_log_returns ** 2).mean())) - 1
            else:
                mean = self.mean_returns(geometric=False)
                excess_returns = self.returns - mean
                downside_returns = excess_returns[excess_returns < 0]
                downside_dev = np.exp(np.sqrt((downside_returns ** 2).mean())) - 1
            
            return downside_dev
        
        except Exception as e:
            logging.exception(f'ERROR calculating Downside standard deviation | {e}')
            return None

    def positive_returns_pct(self):
        try:
            pos = self.returns[self.returns>=0].count()/self.returns.count()
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

    def drawdown(self):
        """
        Calculate the drawdown of a time series of prices.
        
        Parameters:
        prices (pd.Series): A pandas Series containing the price data.
        
        Returns:
        pd.DataFrame: A DataFrame containing the drawdown calculations.
        """
        try:
            # Calculate the cumulative maximum
            cumulative_max = self.balance.cummax()
            # Calculate the drawdown
            drawdown = (self.balance - cumulative_max) / cumulative_max
            
            return drawdown
        except Exception as e:
            logging.exception(f'ERROR calculating Max DD | {e}')
            return None
        
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
            max_drawdown = self.drawdown().min()
            return max_drawdown

        except Exception as e:
            logging.exception(f'ERROR calculating Max DD | {e}')
            return None

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
            # Calculate cumulative maximum to identify the peak
            cumulative_max = self.balance.cummax()
            
            # Calculate drawdowns
            drawdowns = (self.balance - cumulative_max) / cumulative_max
            
            # Identify the period of the maximum drawdown
            max_drawdown_end_date = drawdowns.idxmin()
            max_drawdown_start_date = cumulative_max[:max_drawdown_end_date].idxmax()
            
            # Find the recovery date
            recovery_date = (self.balance[max_drawdown_end_date:] >= self.balance[max_drawdown_start_date]).idxmax()
            
            # Check if the asset has recovered
            if self.balance[recovery_date] < self.balance[max_drawdown_start_date]:
                return "Not Recovered"
            
            # Calculate the number of days to recover
            recovery_days = (recovery_date - max_drawdown_end_date).days
            
            return recovery_days
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
            return None

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
            return None

    def jensen_alpha(self, geometric = True) -> float:
        try:
            if geometric:
                beta = self.beta_alpha(geometric=True)['beta']
                bm_mean_returns = np.exp(self.bm_log_returns.mean())-1
                mean_returns = self.mean_returns(geometric=True)
            else:
                beta = self.beta_alpha(geometric=False)['beta']
                bm_mean_returns = self.bm_returns.mean()
                mean_returns = self.mean_returns(geometric=False)
                
            bm_excess_return = bm_mean_returns - self.rf
            jensen_alpha = mean_returns - (self.rf + beta*bm_excess_return)
            return jensen_alpha
        
        except Exception as e:
            logging.exception(f'ERROR calculating Jensen Alpha | {e}')
            return None