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


class AgingEffect(object):
    # language=rst
    """
    Abstract base class for aging effect of memristor.
    """
    
    def __init__(
            self,
            file,
            dictionary,
            **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.

        :param file: Aging effect data file.
        :param dictionary: Memristor device parameters.
        """
        data = pd.DataFrame(pd.read_excel(
            file,
            sheet_name=0,
            header=None,
            names=[
                'mem_t',
                'G_on',
                'G_off'
            ]
        ))

        self.mem_t = data['mem_t']
        self.points = len(self.mem_t)
        self.G_on = data['G_on']
        self.G_off = data['G_off']

        self.sampling_rate = dictionary['sampling_rate']
        if self.sampling_rate is None:
            self.sampling_rate = 1
        G_init_num = 0
        self.G_on_init = np.median(self.G_on[G_init_num:G_init_num + self.sampling_rate])
        self.G_off_init = np.median(self.G_off[G_init_num:G_init_num + self.sampling_rate])

    def equation_1_log(self, mem_t, r):
        # language=rst
        """
        Equation 1(taking the logarithm) for fitting aging effect.

        :param mem_t: Parameter conductance.
        :param r: Parameter r.
        """
        return np.log(self.G_0) + mem_t * np.log(1 - r)

    def equation_2(self, mem_t, k):
        # language=rst
        """
        Equation 2 for fitting aging effect.

        :param mem_t: Parameter conductance.
        :param k: Parameter k.
        """
        return k * mem_t + self.G_0


    @timer
    def fitting_equation1(self):
        # language=rst
        """
        Fit aging effect using equation 1.
        """
        self.G_0 = self.G_off_init
        params_off, pconv_off = curve_fit(
            self.equation_1_log,
            self.mem_t,
            np.log(self.G_off),
            p0=1e-4
        )

        self.G_0 = self.G_on_init
        params_on, pconv_on = curve_fit(
            self.equation_1_log,
            self.mem_t,
            np.log(self.G_on),
            p0=1e-4
        )

        Aging_off = params_off[0]
        Aging_on = params_on[0]

        return Aging_off, Aging_on


    @timer
    def fitting_equation2(self):
        # language=rst
        """
        Fit aging effect using equation 2.
        """
        self.G_0 = self.G_off_init
        params_off, pconv_off = curve_fit(
            self.equation_2,
            self.mem_t,
            self.G_off,
        )
        
        self.G_0 = self.G_on_init
        params_on, pconv_on = curve_fit(
            self.equation_2,
            self.mem_t,
            self.G_on,
        )

        Aging_off = params_off[0]
        Aging_on = params_on[0]

        return Aging_off, Aging_on
        