import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
class buildIndex(object):
    def __init__(self, start_value, end_date, split=0.8):
        self.start_value = start_value
        self.split = split
        self.end_date = end_date  ##str
    def load_sp_500_data(self):
        df500 = pd.read_csv('sp500.csv')
        past_size = int(df500.shape[0] * self.split)
        start_day = past_size+1  #by loc
        sub_df500 = df500.iloc[start_day:, :]
        return sub_df500
    
    def get_total_value(self, df):
        start_price = df.iloc[0,1]
        end_price = df.loc[df['date'] == self.end_date,'SP 500']
        total_return = (end_price - start_price)/start_price
        total_value = self.start_value * (1+total_return)
        shares = self.start_value/start_price
        df['daily_index_value'] = shares * df.loc[:,'SP 500']
        return total_return
        
        
        
    def plot_portfolio(self,df):
        fig, ax = plt.subplots()
        df.date = df.date.apply(lambda x: datetime.strptime(x, '%m/%d/%Y') )
        ax.plot(df.date,df['daily_index_value'],label = 'sp500')
        
        plt.legend()
        plt.savefig('index_value.png')
    
    
    
        
if __name__ == "__main__":
    s = buildIndex(10000, '7/1/2021')
    df = s.load_sp_500_data()
    total_return = s.get_total_value(df)
    print(total_return)
    s.plot_portfolio(df)