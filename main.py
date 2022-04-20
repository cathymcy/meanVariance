import os
import sys
from data_loader import DataLoader
from mean_variance_portfolio import MeanVariance
import copy
import pandas as pd
from build_index import buildIndex
def my_port_return():
    loader = DataLoader(10, ['open', 'high', 'low', 'close'])
    
    x_train, x_test = loader.load_data_excel('test.xlsx',  0.8)
    loader.my_portfolio_weights(x_train)
    my_port_return = loader.my_portfolio_return(x_test, end_date='2021-07-01',plot=True)
    print("my port return is: ", my_port_return)
    
def build_index():
    s = buildIndex(10000, '7/1/2021')
    df = s.load_sp_500_data()
    total_return = s.get_total_value(df)
    print("index return is: ",total_return)
    s.plot_portfolio(df)
    
if __name__ == "__main__":
    my_port_return()
    build_index()
    
    
    
    
    
    