import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import scipy.optimize as solver
class MeanVariance(object):
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.num_stocks = len(self.mean)
        
    def construct_frontier(self):
        return_list = []
        volatility_list = []
        for _ in range(2000):
            w = np.random.rand(self.num_stocks)
            w /= sum(w)
            port_mean, port_cov = self.port_mean_var(w)
            return_list.append(port_mean)
            volatility_list.append(np.sqrt(port_cov))
        plt.plot(volatility_list, return_list, 'ro')
        plt.show()
    
    def port_mean_var(self, w):
        port_mean = sum(self.mean*w)
        port_var = reduce(np.dot, [w, self.cov, w.T])
        return port_mean, port_var
    
    def solve_weights(self, risk_aversion=1):
        '''
        :param risk_aversion:
        :return: return optimized weights (list of ticker)
        '''
        def fitness(w, risk_aversion):
            port_mean, port_var = self.port_mean_var(w)
            mean_variance_util = port_mean - 1/2 * risk_aversion * port_var
            return 1/mean_variance_util
        
        w = np.ones([self.num_stocks]) / self.num_stocks
        b_ = [(0., 1.) for i in range(self.num_stocks)] # weights between 0%..100%, No leverage, no shorting
        c_ = ({'type': 'eq', 'fun': lambda w: sum(w) - 1.}) # Sum of weights = 100%
        optimized = solver.minimize(fitness, w, (risk_aversion),
                                            method='SLSQP', constraints=c_, bounds=b_)
        if not optimized.success:
            raise BaseException(optimized.message)
        return optimized.x  # Return optimized weights
    