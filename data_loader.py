import quandl
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import xlrd
import copy
import matplotlib.pyplot as plt
from datetime import datetime
from mean_variance_portfolio import MeanVariance
quandl.ApiConfig.api_key="vrJkt5c3B5qJRExuYYPg"

class DataLoader:
    def __init__(self, seq_len, select_features):
        self.seq_len = seq_len
        self.select_features = select_features
        self.ticker_list = []
    def load_data(self, tickers, start, end, split, min_days = 0):
        x_train = pd.DataFrame()
        x_test = pd.DataFrame()
        
        for ticker in tickers:
            table = quandl.get_table('SHARADAR/SEP',
                                     qopts={"columns":['ticker' ,'date','open','high','low','close','volume','closeadj']},
                                     date={'gte': start, 'lte': end},
                                     ticker=ticker)

          
            print(table.shape[0], ticker)
            if table.shape[0] < min_days:
                raise("Not Enough Data for tick {}".format(ticker))
            else:
                training_size = int(table.shape[0] * split)
                if training_size <= self.seq_len:
                    print("not enough data")
                    continue
                df = table[['date', 'closeadj']]
                df.date = df.date.apply(lambda x: x.date())
                df = df.set_index('date')
                df.sort_index(inplace=True)  # reverse, table[0] is the earliest day
                
                x_train_per_ticker = df[:training_size]
                x_test_per_ticker = df[training_size+1:]
                x_train_per_ticker.rename(columns={'closeadj':ticker},inplace=True)
                x_test_per_ticker.rename(columns={'closeadj':ticker},inplace=True)
                if x_train.empty:
                    x_train = x_train_per_ticker
                    x_test = x_test_per_ticker
                else:
                    x_train = x_train.join(x_train_per_ticker)
                    x_test = x_test.join(x_test_per_ticker)
        x_train.fillna(method='backfill')
        return x_train, x_test
    
    def load_data_excel(self, file, split):
        x_train = pd.DataFrame()
        x_test = pd.DataFrame()
        xls = xlrd.open_workbook(file,on_demand=True)
        #xls = pd.read_excel(file, engine='openpyxl')
        for ticker in xls.sheet_names():
            self.ticker_list.append(ticker)
            table = pd.read_excel(file, ticker)
            training_size = int(table.shape[0] * split)
            if training_size <= self.seq_len:
                print("not enough data")
                continue
            df = table[['date', 'closeadj']]
            df.date = df.date.apply(lambda x: x.date())
            df = df.set_index('date')
            df.sort_index(inplace=True)  # reverse, table[0] is the earliest day

            x_train_per_ticker = df[:training_size]
            x_test_per_ticker = df[training_size + 1:]
            x_train_per_ticker.rename(columns={'closeadj': ticker}, inplace=True)
            x_test_per_ticker.rename(columns={'closeadj': ticker}, inplace=True)
            if x_train.empty:
                x_train = x_train_per_ticker
                x_test = x_test_per_ticker
            else:
                x_train = x_train.join(x_train_per_ticker)
                x_test = x_test.join(x_test_per_ticker)

        x_train = x_train.fillna(method='backfill')
        x_test = x_test.fillna(method='backfill')
        return x_train, x_test
   
    @staticmethod
    def log_return_calculation(df):
        df_log_return = np.log(df) - np.log(df.shift(1))
        return df_log_return
    
    
    
    def my_portfolio_weights(self, x_train):
        df_train_log_return = DataLoader.log_return_calculation(x_train)
        
        mean = df_train_log_return.mean() * 252
        cov = df_train_log_return.cov() * 252
        m = MeanVariance(mean, cov)
        weights = m.solve_weights()
    
        self.update_weights = copy.deepcopy(weights)
        for i in range(len(self.ticker_list)):
            if self.update_weights[i] < 10 ** (-10):
                self.update_weights[i] = 0
    
        ticker_weights_map = dict(zip(self.ticker_list, self.update_weights))
        df_ticker_weights = pd.DataFrame(ticker_weights_map.items(), columns=['ticker', 'weight'])
        self.df_hold = df_ticker_weights.loc[~(df_ticker_weights['weight'] == 0)]
        self.df_hold.to_csv('df_hold.csv')

    def my_portfolio_return(self,x_test, end_date, port_start_value=10000, plot=False):
        '''
        return is calculated in the test dataframe
        '''
        my_portfolio_end_value = 0
        x_test.index = pd.to_datetime(x_test.index)
        x_test['my_port_value'] = 0
        
        for i, row in self.df_hold.iterrows():
            i_ticker = row['ticker']
            i_weight = row['weight']
            i_end_price = x_test.loc[end_date,i_ticker]
            i_start_price = x_test.iloc[0][i_ticker]
            i_shares = i_weight*port_start_value/i_start_price
            my_portfolio_end_value += i_shares*i_end_price
            x_test['my_port_value'] += i_shares*x_test.loc[:,i_ticker]
        my_port_return = (my_portfolio_end_value - port_start_value)/port_start_value
        
        
        if plot:
            fig, ax = plt.subplots()
            ax.plot(x_test.index, x_test['my_port_value'], label='my_portfolio')

            plt.legend()
            plt.savefig('my_port_value.png')
        return my_port_return
    
    
        