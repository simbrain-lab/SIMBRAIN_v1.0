import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import fsolve

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


class RetentionLoss(object):
    # language=rst
    """
    Abstract base class for retention loss of memristor.
    """
    
    def __init__(
            self,
            file,
            **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.

        :param file: Retention loss data file.
        """
        data = pd.DataFrame(pd.read_excel(
            file,
            sheet_name=0,
            header=None,
            names=[
                'Time(s)',
                'Conductance(S)'
            ]
        ))

        self.time = np.array(data['Time(s)'])
        self.conductance = np.array(data['Conductance(S)'])
        self.delta_t = data['Time(s)'][1] - data['Time(s)'][0]
        self.w_init = self.conductance[0]
        self.points = len(self.time)


    def retention_loss(self, time, k, beta):
        # language=rst
        """
        Simplified equation for fitting retention loss.

        :param time: Parameter t.
        :param k: Parameter k.
        :param beta: Parameter beta.
        """
        internal_state = np.zeros(self.points)
        internal_state[0] = self.w_init

        for i in range(self.points - 1):
            internal_state[i + 1] = internal_state[i] - self.delta_t * k * internal_state[i] * (
                    (time[i]) ** (beta - 1))

        return internal_state
    
    
    def rentention_loss_1(self, time, tau_reciprocal):
        # language=rst
        """
        Original equation for fitting retention loss.

        :param time: Parameter t.
        :param tau_reciprocal: Parameter tau.
        """
        internal_state = np.zeros(self.points)
        internal_state[0] = self.w_init

        for i in range(self.points - 1):
            internal_state[i + 1] = internal_state[i] - self.delta_t * tau_reciprocal * internal_state[i] * time[i]

        return internal_state
            

    def equation(self, tau_reciprocal):
        # language=rst
        """
        Equation for calculating the simplifying error.

        :param tau_reciprocal: Parameter tau.
        """
        return self.beta * np.power(tau_reciprocal, self.beta) - self.k


    @timer
    def fitting(self):
        # language=rst
        """
        Fit retention loss.
        """
        self.w_init = self.conductance[0] * 1.002
        params, pconv = curve_fit(
            self.retention_loss,
            self.time,
            self.conductance
        )
        # print(params, pconv)
        k, beta = params[0], params[1]
        if k > 0:
            tau_reciprocal = np.power(k/beta, 1/beta) 
        else:
            params, pconv = curve_fit(
                self.retention_loss,
                self.time,
                self.conductance
            )
            tau_reciprocal = params[0]
            beta = 1            

        return tau_reciprocal, beta
