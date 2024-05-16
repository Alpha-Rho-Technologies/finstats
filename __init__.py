from finstats.src.core import *
import datetime as dt
import calendar

class sbs:
    def __init__(self,balance:pd.Series,bm_balance:pd.Series,start_date:dt.date=None,end_date:dt.date=None) -> None:
        '''
        Initialize an instance of the single balance stats (sbs) class.

        This class is designed to analyze and retrieve financial statistics based on a given balance data series. It compares the provided balance data against a benchmark balance over a specified date range.

        Parameters:
        balance (pd.Series): A pandas Series object representing the balance data to be analyzed.
        start_date (dt.date): The starting date of the period for which the balance statistics are to be calculated.
        end_date (dt.date): The ending date of the period for which the balance statistics are to be calculated.
        bm_balance (pd.Series): A pandas Series object representing the benchmark balance data for comparison.
        '''

        try:
            # set dates:
            if start_date is None and end_date is None:
                start_date,end_date = default_dates(balance)

            raw_data = [bm_balance,balance]
            formatted_data = {}
            for data in raw_data:
                name = data.name

                # Adjust date:
                check_data_index(data)
                data = data.loc[start_date:end_date]
                formatted_data[name] = data
            
            self.balance = formatted_data[balance.name]
            self.bm_balance = formatted_data[bm_balance.name]
        
        except Exception as e:
            logging.exception(f'ERROR initializing strategy stats | {e}')

    def get_stats(self,freq=str,annual_rf = 0.022) -> dict:
        '''
        Calculate and return various financial statistics based on the initialized balance data.

        This method computes key financial metrics such as return rates, volatility, and risk-adjusted returns over the specified frequency. The calculations take into account a predefined annual risk-free rate.

        Parameters:
        freq (str): The frequency at which the statistics should be calculated. Common values include 'daily', 'monthly', or 'yearly'.
        annual_rf (float, optional): The annual risk-free rate used in the calculation of risk-adjusted returns. Default value is 0.022 (or 2.2%).

        Returns:
        dict: A dictionary containing the calculated financial statistics. Keys in the dictionary might include metrics such as 'total_return', 'volatility', and 'sharpe_ratio', among others, depending on the implementation.

        Note:
        The method assumes that the class has been properly initialized with the necessary balance data and benchmark balance data.
        '''
        try:
            # Standarize Rf:
            rf = get_rf(rf=annual_rf,stats_freq=freq)

            # Initialize stats object:
            self.stats = fin_stats(balance = self.balance,
                                    bm_balance = self.bm_balance,
                                    stats_freq = freq,rf=rf)

            # Basic Stats:
            mean = self.stats.mean_returns()
            std = self.stats.returns_standard_deviation()

            # Misc:
            excess_return = mean - rf
            downside_dev = self.stats.downside_deviation()
            pos = self.stats.positive_returns_pct()
            neg = 1 - pos
            linreg = self.stats.beta_alpha()
            
            # Risk-adjusted performance ratios:
            sharpe = self.stats.sharpe_ratio()
            sortino = self.stats.sortino_ratio()
            
            # Investment Risk Measures:
            var_99 = mean - std * 2.56
            es_99 = self.stats.es()
            max_loss = self.stats.min_return()
            max = self.stats.max_return()
            losing_streak = self.stats.losing_streak()
            max_dd = self.stats.max_dd()
            recovery = self.stats.recovery()
            corr = self.stats.correlation()

            calmar_ratio = excess_return/abs(max_loss)

            info = {
                'Stats Since': str(self.balance.index[0].date()),
                'Geometric Mean Return': mean,
                'Standard Deviation': std,
                'Downside Standard Deviation':downside_dev,
                'Sharpe Ratio':sharpe,
                'Sortino Ratio':sortino,
                'Calmar Ratio':calmar_ratio,
                'Max DD':max_dd,
                'Max Return': max,
                'Min Return': max_loss,
                'VAR 99':var_99,
                'ES 99': es_99,
                'Positive %':pos,
                'Negative %':neg,
                'Max Losing Streak': losing_streak,
                'Recovery Max DD': recovery,
                'Correlation': corr
                }
                
            # Stats only relevant to strategy:
            info['Information Ratio'] = self.stats.info_ratio()
            info['Beta'] = linreg['beta']
            info['Alpha'] = linreg['alpha']
            info['Jensen Alpha'] = self.stats.jensen_alpha(rf=rf)
            
            return info
        
        except Exception as e:
            logging.exception(f'ERROR Retriving balance stats | {e}')    
        
    def df(self,freq=str):
        '''
        return data series with strategy stats
        '''
        try:
            strat_stats = self.get_stats(freq=freq)
            return pd.Series(strat_stats).rename(f'{freq} Stats')
        except Exception as e:
            logging.exception(f'ERROR Retriving Stats df | {e}')

    def returns_by_month(self,dropnan:bool):
        '''
        Returns monthly returns by year and month
        '''
        try:
            monthly_balance = self.balance.groupby(pd.Grouper(freq='M')).last()
            df = monthly_balance.pct_change(fill_method=None).to_frame()
            df.index = pd.to_datetime(df.index)

            # set year and month columns
            df['Year'] = df.index.year
            df['month'] = df.index.month

            # create a pivot table to get the month-on-month price change
            pivot = pd.pivot_table(df,index='Year', columns='month', aggfunc='first')
            pivot.columns = list(calendar.month_name)[1:]

            # Remove non-complete years:
            if dropnan:
                pivot.dropna(axis=0,inplace=True)

            # Add Yearly returns:
            pivot['Yearly Returns'] = np.exp(np.log(1+pivot).sum(axis=1))-1

            # Add Monthly AVG:
            pivot.loc['Geometric Average'] = np.exp(np.log(1+pivot).mean())-1
            return round(pivot,4)
        
        except Exception as e:
            logging.exception(f'ERROR Retriving Returns by month | {e}')
    
    def rolling_correlation(self,periods:int,pct=True):
        bal_pct = self.balance.pct_change()
        if pct == True:
            bm_pct = self.bm_balance.pct_change()
        else:
            bm_pct = self.bm_balance.diff()
        
        return bal_pct.rolling(periods).corr(bm_pct)
    
    def rolling_alpha(self,period):
        alphas = {}
        df = pd.concat([self.bm_balance.pct_change(),self.balance.pct_change()],axis=1).dropna()
        for data in df.rolling(period):
            if len(data) >= period:
                date = data.index[-1]
                alpha = lr(x=data.iloc[:,0],y=data.iloc[:,1])[1]
                alphas[date] = alpha

        return pd.Series(alphas)

class mbs:
    def __init__(self,asset_price_data=pd.DataFrame,bm_data = pd.Series,start_date=dt.date,end_date=dt.date) -> None:
        '''
        Initialize an instance of the mbs (Multiple Balance Stats) class.

        This class is designed for calculating various financial statistics across multiple assets. It uses provided asset price data, compares it against a benchmark data series, and performs analysis within a specified date range.

        Parameters:
        asset_price_data (pd.DataFrame): A pandas DataFrame containing the asset price data. Each column in the DataFrame represents a different asset, with rows corresponding to different dates.
        start_date (dt.date): The starting date for the period over which the financial statistics are to be calculated.
        end_date (dt.date): The ending date for the period over which the financial statistics are to be calculated.
        bm_data (pd.Series): A pandas Series representing the benchmark data for comparison with the asset price data.

        Returns:
        None: This constructor method initializes the class instance but does not return any value.

        Note:
        The asset price data and the benchmark data should be aligned in terms of dates, with each row corresponding to the same date across all data series.
        '''
        raw_apd = asset_price_data.loc[start_date:end_date]
        self.apd = format_raw_data(raw_data=raw_apd)
        self.start_date = start_date
        self.end_date = end_date
        self.bm_df = bm_data
    
    def stats_df(self,freq):
        '''
        Calculate and return a DataFrame of financial statistics for multiple assets at the specified frequency.

        This method processes the asset price data initialized in the mbs class instance and computes key financial metrics for each asset at the given frequency. The statistics are calculated based on the price data and benchmark data provided during class initialization.

        Parameters:
        freq (str): The frequency at which the financial statistics should be calculated. This parameter determines the time intervals for the calculations. Common values include 'B', 'D', 'M' or 'Y'.

        Returns:
        pd.DataFrame: A pandas DataFrame where each column represents a different financial metric calculated for each asset, and each row corresponds to the specified frequency intervals. The specific metrics included in the DataFrame will depend on the implementation but may include returns, volatility, and other relevant financial measures.

        Note:
        This method assumes that the mbs class has been properly initialized with the necessary asset price data and benchmark data.
        '''
        stats = []
        for asset in self.apd.columns:
            balance = self.apd[asset]
            stats_series = sbs(balance=balance,
                        start_date=self.start_date,
                        end_date=self.end_date,
                        bm_balance=self.bm_df).df(freq=freq)
            stats.append(stats_series.rename(asset))
        
        return pd.concat(stats,axis=1)
    
    def indexed_returns(self,freq='B'):
        '''
        Calculate and return indexed returns for the asset prices over the specified frequency.

        This method computes the percentage change in asset prices at the given frequency, then converts these to indexed returns. The indexed returns are normalized to a base value of 100 at the start date. This allows for an easy comparison of asset performance over time.

        Parameters:
        freq (str, optional): The frequency at which the returns are calculated. Defaults to 'B' (business daily). Other common frequencies include 'W' (weekly), 'M' (monthly), 'Q' (quarterly), etc.

        Returns:
        pd.DataFrame: A pandas DataFrame containing the indexed returns of the assets. Each column represents an asset, and each row corresponds to a date. The DataFrame is sorted by date, with NaN values dropped, and the returns are rounded to three decimal places.

        Note:
        The method uses the asset price data (`apd`) that should already be present in the class instance. It assumes that the `apd` data is a pandas DataFrame with dates as the index and asset prices in the columns.
        '''
    
        pct_df = self.apd.resample(freq).last().pct_change(fill_method=None)
        log_pct = np.log(1+pct_df)
        prices_indexed = 100*np.exp(log_pct.cumsum())
        
        # Add initial index:
        prices_indexed.loc[self.start_date] = 100
        prices_indexed.index = pd.to_datetime(prices_indexed.index)
        return round(prices_indexed.sort_index(),2).dropna()
    
def asset_perf_contribution(start_date, end_date, asset_price_data=pd.DataFrame, portfolio=pd.Series):
    '''
    Performance contribution by asset
    '''
    # Calculate percentage change
    pct_change = asset_price_data.loc[start_date:end_date].pct_change(fill_method=None)
    
    # Calculate log returns
    log_returns = np.log(1 + pct_change)
    
    # Calculate cumulative returns
    cumulative_returns = np.exp(log_returns.cumsum()) - 1
    
    # Calculate asset contribution
    asset_contribution = cumulative_returns.iloc[-1].loc[portfolio.index] * portfolio
    
    return asset_contribution

def seasonality(price_data:pd.Series):
    """
    Analyzes the seasonality in financial price data by calculating and summarizing 
    the monthly percentage changes.

    Parameters:
    price_data (pd.Series): A Pandas Series object with timestamps as index and 
                            financial prices as values.

    Returns:
    pd.DataFrame: A DataFrame where each row corresponds to a month, and columns 
                  contain descriptive statistics of the monthly percentage changes.
    """
    # Ensure price_data is a Pandas Series
    if not isinstance(price_data, pd.Series):
        raise ValueError("price_data must be a pandas Series")

    # Calculate monthly percentage change
    df_pct = np.log(1 + price_data.resample('M').last().pct_change())

    # Group by month and calculate descriptive statistics
    monthly_stats = np.exp(df_pct.groupby(df_pct.index.month).describe()) - 1

    # Replace numerical index with month names
    month_names = [calendar.month_name[i] for i in monthly_stats.index]
    monthly_stats.index = month_names

    return monthly_stats