import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('%s cost time: %.3f s' % (func.__name__, time_spend))
        return result

    return func_wrapper


def gaussian(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


class StuckAtFault:
    # language=rst
    """
    Abstract base class for stuck at fault of memristor crossbar.
    """
    
    def __init__(
            self,
            file,
            **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.

        :param file: SAF data file.
        """
        self.data = pd.DataFrame(pd.read_excel(
            file,
            sheet_name=0,
            header=None,
            index_col=0
        )).T


    @timer
    def pre_deployment_fitting(self):
        # language=rst
        """
        Fit pre-deployment SAF.
        """
        SAF_lambda = np.count_nonzero(self.data[0]) / self.data.shape[0]
        SA_0 = np.count_nonzero(self.data[0] == -1) / self.data.shape[0]
        SA_1 = np.count_nonzero(self.data[0] == 1) / self.data.shape[0]
        SAF_ratio = np.array(SA_0) / np.array(SA_1) 

        return SAF_lambda, SAF_ratio


    @timer
    def post_deployment_fitting(self):
        # language=rst
        """
        Fitting post-deployment SAF.
        """
        if self.data.shape[1] > 1:
            mem_t = self.data.columns[1] - self.data.columns[0]
            SAF = []
            for i in range(self.data.shape[1]):
                SAF.append(np.count_nonzero(self.data[mem_t * i]) / self.data.shape[0])
            z1 = np.polyfit(self.data.columns.values, np.array(SAF), 1)
            # p1 = np.poly1d(z1)
            # print(p1)
            SAF_delta =z1[0]
        else:
            SAF_delta = 0
            
        return SAF_delta
