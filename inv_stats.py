from fin_stats.stats_funcs import *

class strategy_stats:
    def __init__(self,balance:pd.Series,start_date:dt.date,end_date:dt.date,bm_balance:pd.Series) -> None:
        '''
        Retrive relevant financial stats on a given balance data series.
        '''
        try:
            # Ensure Dtatime index:
            bm_name = bm_balance.name
            self.balance_name = balance.name
            raw_data = {
                bm_name:bm_balance,
                self.balance_name:balance
            }

            formatted_data = {}
            for name,data in raw_data.items():
                # Adjust date:
                data.index = pd.to_datetime(data.index)
                data = data.loc[start_date:end_date]
                formatted_data[name] = data
            

            self.balance = formatted_data[self.balance_name]
            self.bm_balance = formatted_data[bm_name]
        
        except Exception as e:
            logging.exception(f'ERROR initializing strategy stats | {e}')

    def get_stats(self,freq=str,annual_rf = 0.022) -> dict:
        try:
            # Standarize Rf:
            rf = get_rf(rf=annual_rf,freq=freq)

            # Adjust frequency balance:
            bal_adj = self.balance.resample(freq).last()
            bm_bal_adj = self.bm_balance.resample(freq).last()

            mean = cal_geo_mean(balance=bal_adj)
            std = cal_geo_std(balance=bal_adj)
            excess_return = mean-rf
            sharpe = excess_return/std

            downside_dev = calc_downside_dev(balance=bal_adj)
            sortino = excess_return/downside_dev
            var_99 = mean-std*2.56
            es_99 = calc_es(balance=bal_adj)
            max_loss = calc_min_return(balance=bal_adj)
            max = calc_max_return(balance=bal_adj)
            pos = calc_pos_returns_pct(balance=bal_adj)
            neg = 1-pos
            losing_streak = calc_losing_streak(self.balance)
            max_dd = calc_max_dd(self.balance)
            recovery = calc_recovery(self.balance)
            corr = calc_corr(balance=bal_adj, bm_balance= bm_bal_adj)

            calmar_ratio = excess_return/abs(max_loss)

            stats = {
                'Stats Since': str(bal_adj.index[0].date()),
                'Geometric Mean Return': mean,
                'STD': std,
                'Downside STD':downside_dev,
                'Sharpe Ratio':sharpe,
                'Sortino Ratio':sortino,
                'Calmar Ratio':calmar_ratio,
                'Max DD':max_dd,
                'Max Return': max,
                'Max Loss': max_loss,
                'VAR 99':var_99,
                'ES 99': es_99,
                'Positive %':pos,
                'Negative %':neg,
                'Max Losing Streak': losing_streak,
                'Recovery Max DD': recovery,
                'Correlation': corr
                }
                
            # Stats only relevant to strategy:
            stats['Information Ratio'] = calc_info_ratio(balance=bal_adj,
                                                            bm_balance=bm_bal_adj)

            linreg = calc_beta_alpha(balance=bal_adj,
                                    bm_balance=bm_bal_adj)
            stats['Beta'] = linreg['beta']
            stats['Alpha'] = linreg['alpha']
            stats['Jensen Alpha'] = calc_jensen_alpha(balance=bal_adj,
                                                            bm_balance=bm_bal_adj,
                                                            rf=rf,beta=linreg['beta'])
            return stats
        
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
            pivot.loc['Geometric Average'] = np.exp(np.log(1+pivot).mean())-1
            return round(pivot,6)
        
        except Exception as e:
            logging.exception(f'ERROR Retriving Returns by month | {e}')

class multiple_strategy:
    def __init__(self,asset_price_data=pd.DataFrame,start_date=dt.date,end_date=dt.date,bm_data = pd.Series) -> None:
        self.apd = asset_price_data.loc[start_date:end_date].dropna(axis=1)
        self.start_date = start_date
        self.end_date = end_date
        self.bm_df = bm_data
    
    def stats_df(self,freq):
        stats = []
        for asset in self.apd.columns:
            balance = self.apd[asset]
            stats_series = strategy_stats(balance=balance,
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