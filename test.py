import unittest
from data_loader import DataLoader
class TestMethods(unittest.TestCase):
    def test_data_loader(self):
        loader = DataLoader(10, ['open', 'high', 'low', 'close'])
        tickers = ['MSFT', 'AMZN', 'FB', 'GOOGL', 'GOOG', 'BRK.B', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'UNH', 'HD', 'PG',
                   'MA', 'PYPL', 'DIS', 'ADBE', 'BAC']
    
        x_train, x_test = loader.load_data(tickers, '2012-01-01', '2021-7-1', 0.8)
        assert len(x_test)!=0
        
    def test_load_data_excel(self):
        loader = DataLoader(10, ['open', 'high', 'low', 'close'])
        x_train, x_test = loader.load_data_excel(file='test.xlsx', split=0.8)
        df_train_log_return = DataLoader.log_return_calculation(x_train)
        df_test_log_return = DataLoader.log_return_calculation(x_test)
        assert df_train_log_return.shape[1]==10
        assert df_test_log_return.shape[1]==10