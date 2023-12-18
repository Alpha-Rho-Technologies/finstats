from finstats.src.core import *
import calendar

class balance_stats:
    def __init__(self,balance:pd.Series,start_date:dt.date,end_date:dt.date,bm_balance:pd.Series) -> None:
        '''
        Retrive relevant financial stats on a given balance data series.
        '''
        try:
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
        try:
            # Standarize Rf:
            rf = get_rf(rf=annual_rf,stats_freq=freq)

            # Initialize stats object:
            stats = fin_stats(balance = self.balance,
                              bm_balance = self.bm_balance,
                              stats_freq = freq)

            # Basic Stats:
            mean = stats.mean_returns()
            std = stats.returns_standard_deviation()

            # Misc:
            excess_return = mean - rf
            downside_dev = stats.downside_deviation()
            pos = stats.positive_returns_pct()
            neg = 1 - pos
            linreg = stats.beta_alpha()
            
            # Risk-adjusted performance ratios:
            sharpe = excess_return/std
            sortino = excess_return/downside_dev
            
            # Investment Risk Measures:
            var_99 = mean - std * 2.56
            es_99 = stats.es()
            max_loss = stats.min_return()
            max = stats.max_return()
            losing_streak = stats.losing_streak()
            max_dd = stats.max_dd()
            recovery = stats.recovery()
            corr = stats.correlation()

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
            info['Information Ratio'] = stats.info_ratio()
            info['Beta'] = linreg['beta']
            info['Alpha'] = linreg['alpha']
            info['Jensen Alpha'] = stats.jensen_alpha(rf=rf)
            
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

    def returns_by_month(self):
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

            # Add Yearly returns:
            pivot['Yearly Returns'] = np.exp(np.log(1+pivot).sum(axis=1))-1

            # Add Monthly AVG:
            pivot.loc['Annualized Returns'] = np.exp(np.log(1+pivot).mean())-1
            return round(pivot,6)
        
        except Exception as e:
            logging.exception(f'ERROR Retriving Returns by month | {e}')

class mbs:
    def __init__(self,asset_price_data=pd.DataFrame,start_date=dt.date,end_date=dt.date,bm_data = pd.Series) -> None:
        '''
        Multiple Balance Stats calculator
        '''
        raw_apd = asset_price_data.loc[start_date:end_date]
        self.apd = format_raw_data(raw_data=raw_apd)
        self.start_date = start_date
        self.end_date = end_date
        self.bm_df = bm_data
    
    def stats_df(self,freq):
        stats = []
        for asset in self.apd.columns:
            balance = self.apd[asset]
            stats_series = balance_stats(balance=balance,
                        start_date=self.start_date,
                        end_date=self.end_date,
                        bm_balance=self.bm_df).df(freq=freq)
            stats.append(stats_series.rename(asset))
        
        return pd.concat(stats,axis=1)
    
    def indexed_returns(self,freq='M'):
    
        pct_df = self.apd.resample(freq).last().pct_change(fill_method=None)
        
        prices_indexed = 100*np.exp(np.log(1+pct_df).cumsum())
        
        # Add initial index:
        prices_indexed.loc[self.start_date] = 100
        prices_indexed.index = pd.to_datetime(prices_indexed.index)
        return round(prices_indexed.sort_index(),3).dropna()
    
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

def seasonality(price_data=pd.DataFrame,reb_freq=int):
    df_pct = np.log(1+price_data.resample(f'{reb_freq}M').last().pct_change(fill_method=None))
    df = np.exp(df_pct.groupby(df_pct.index.month).describe())-1
    
    months = []
    for month in df.index:
        months.append(list(calendar.month_name)[month])
    df.index = months
    return df