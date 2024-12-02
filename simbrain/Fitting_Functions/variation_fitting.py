import math
import numpy as np
import pandas as pd
from scipy.stats import norm


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


class Variation(object):
    # language=rst
    """
    Abstract base class for variations of memristor crossbar.
    """
    
    def __init__(
            self,
            file,
            G_off_list: np.array,
            G_on_list: np.array,
            P_off_list: np.array,
            P_on_list: np.array,
            dictionary: dict = {},
            **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.

        :param file: Conductance data file.
        :param G_off_list: The list of parameter G_off from multiple devices
        :param G_on_list: The list of parameter G_on from multiple devices
        :param P_off_list: The list of parameter P_off from multiple devices
        :param P_on_list: The list of parameter P_on from multiple devices
        :param dictionary: Memristor device parameters.
        """
        # Read excel
        self.data = pd.DataFrame(pd.read_excel(
            file,
            sheet_name='Sheet1',
            header=None,
            index_col=None
        ))
        self.data.columns = ['Pulse Voltage(V)', 'Read Voltage(V)'] + list(self.data.columns[2:] - 2)

        # Read parameters
        self.J1 = 1
        self.v_off = dictionary['v_off']
        self.v_on = dictionary['v_on']
        self.G_off = dictionary['G_off']
        self.G_on = dictionary['G_on']
        self.alpha_off = dictionary['alpha_off']
        self.alpha_on = dictionary['alpha_on']
        self.P_off = dictionary['P_off']
        self.P_on = dictionary['P_on']        
        self.k_off = dictionary['k_off']
        self.k_on = dictionary['k_on']
        self.delta_t = dictionary['delta_t']
        self.duty_ratio = dictionary['duty_ratio']

        # Read data
        self.V_write = np.array(self.data['Pulse Voltage(V)'])
        self.points_r = np.sum(self.V_write > 0)
        self.points_d = np.sum(self.V_write < 0)
        self.V_write_r = self.V_write[:self.points_r]
        self.V_write_d = self.V_write[self.points_r:]
        self.read_voltage = np.array(self.data['Read Voltage(V)'])[0]

        # Set default parameters
        self.device_nums = self.data.shape[1] - 2
        self.G_off_variation = G_off_list
        self.G_on_variation = G_on_list
        if self.G_off is None:
            self.G_off = np.mean(self.G_off_variation)
        if self.G_on is None:
            self.G_on = np.mean(self.G_on_variation)

        self.P_off_variation = P_off_list
        self.P_on_variation = P_on_list

        if None in [self.P_off, self.P_on]:
            self.P_off = np.mean(self.P_off_variation)
            self.P_on = np.mean(self.P_on_variation)


    def Memristor_conductance_model(self, P_off, P_on, initial, V_write):
        # language=rst
        """
        The Memristor model.

        :param P_off: Memristor parameter P_off.
        :param P_on: Memristor parameter P_on.
        :param initial: The initial value of the internal state variable.
        :param V_write: The voltage of the input write pulse.
        """
        points = len(V_write)
        # initialization
        internal_state = [0 for i in range(points)]
        internal_state[0] = initial
        # conductance change
        for i in range(points - 1):
            if V_write[i + 1] > self.v_off and V_write[i + 1] > 0:
                delta_x = (
                        self.k_off
                        * ((V_write[i + 1] / self.v_off - 1) ** self.alpha_off)
                        * self.J1
                        * ((1 - internal_state[i]) ** P_off)
                )
                internal_state[i + 1] = internal_state[i] + self.delta_t * self.duty_ratio * delta_x

            elif V_write[i + 1] < 0 and V_write[i + 1] < self.v_on:
                delta_x = (
                        self.k_on
                        * ((V_write[i + 1] / self.v_on - 1) ** self.alpha_on)
                        * self.J1
                        * (internal_state[i] ** P_on)
                )
                internal_state[i + 1] = internal_state[i] + self.delta_t * self.duty_ratio * delta_x

            else:
                delta_x = 0
                internal_state[i + 1] = internal_state[i]

            if internal_state[i + 1] < 0:
                internal_state[i + 1] = 0
            elif internal_state[i + 1] > 1:
                internal_state[i + 1] = 1

        return internal_state


    @timer
    def d2d_G_fitting(self):
        # language=rst
        """
        Fit device to device variation of parameter G.
        """
        G_off_cal = (self.G_off_variation - self.G_off) / self.G_off
        G_on_cal = (self.G_on_variation - self.G_on) / self.G_on
        Goff_mu_noise, Goff_sigma = norm.fit(G_off_cal)
        Goff_mu = self.G_off * (1 + Goff_mu_noise)
        Gon_mu_noise, Gon_sigma = norm.fit(G_on_cal)
        Gon_mu = self.G_on * (1 + Gon_mu_noise)

        return Goff_mu, Goff_sigma, Gon_mu, Gon_sigma


    @timer
    def d2d_P_fitting(self):
        # language=rst
        """
        Fit device to device variation of parameter p.
        """
        P_off_cal = (self.P_off_variation - self.P_off)
        P_on_cal = (self.P_on_variation - self.P_on)
        Poff_mu_noise, Poff_sigma = norm.fit(P_off_cal)
        Poff_mu = self.P_off + Poff_mu_noise
        Pon_mu_noise, Pon_sigma = norm.fit(P_on_cal)
        Pon_mu = self.P_on + Pon_mu_noise

        return Poff_mu, Poff_sigma, Pon_mu, Pon_sigma


    @timer
    def c2c_fitting(self, cluster_option='ef'):
        # language=rst
        """
        Fit cycle to cycle variation.

        :param cluster_option: The clustering method used in 4-stage pipeline fitting. Possible values are:
        - 'ef': Divide data into clusters with equal frequency.
        - 'ew': Divide data into clusters with equal width or interval.
        """
        # Conductance from devices
        current_r = np.array(self.data[:])[:self.points_r, 2:]
        current_d = np.array(self.data[:])[self.points_r:, 2:]
        conductance_r = current_r / self.read_voltage
        conductance_d = current_d / self.read_voltage
        # If considering d2d?
        # x_r = ((conductance_r - self.G_on_variation[np.newaxis, :])
        #        / (self.G_off_variation[np.newaxis, :] - self.G_on_variation[np.newaxis, :]))
        # x_d = ((conductance_d - self.G_on_variation[np.newaxis, :])
        #        / (self.G_off_variation[np.newaxis, :] - self.G_on_variation[np.newaxis, :]))
        x_r = (conductance_r - self.G_on) / (self.G_off - self.G_on)
        x_d = (conductance_d - self.G_on) / (self.G_off - self.G_on)
        x_total = np.concatenate((x_r, x_d)).T.flatten()

        # Conductance from models
        x_init_r = np.max([0, np.mean(x_r[0])])
        x_init_d = np.min([1, np.mean(x_d[0])])
        best_mem_x_r = []
        best_mem_x_d = []
        for i in range(self.device_nums):
            # best_mem_x_r.append(np.array(self.Memristor_conductance_model(
            #     self.P_off_variation[i],
            #     self.P_on_variation[i],
            #     x_r[i][0],
            #     self.V_write_r
            # ))[:])
            # best_mem_x_d.append(np.array(self.Memristor_conductance_model(
            #     self.P_off_variation[i],
            #     self.P_on_variation[i],
            #     x_d[i][0],
            #     self.V_write_d
            # ))[:])
            best_mem_x_r.append(np.array(self.Memristor_conductance_model(
                self.P_off,
                self.P_on,
                x_init_r,
                self.V_write_r
            ))[:])
            best_mem_x_d.append(np.array(self.Memristor_conductance_model(
                self.P_off,
                self.P_on,
                x_init_d,
                self.V_write_d
            ))[:])

        best_memx_total = np.concatenate((np.array(best_mem_x_r), np.array(best_mem_x_d)), axis=1).flatten()

        variation_x = abs(best_memx_total - x_total)

        # %% Variation Analysis
        var_x_complex = []
        for i in range(len(best_memx_total)):
            var_x_complex.append((best_memx_total[i], variation_x[i]))


        def takeFirst(ele):
            return ele[0]

        # Sort
        var_x_complex.sort(key=takeFirst)

        # %% Variation Clustering
        x_sorted = np.array([i[0] for i in var_x_complex])
        var_x_sorted = np.array([i[1] for i in var_x_complex])

        if cluster_option == 'ef':
            group_no = max(10, int((len(x_total)/5)**0.5))
            print('Group number:{}'.format(group_no))

            # Group with pulse number
            total_points = (self.points_r + self.points_d) * self.device_nums
            every_points = math.ceil(total_points / group_no)

            x_mean = []
            var_x_median = []
            var_x_average = []
            for i in range(group_no):
                temp_x_mean = np.mean(x_sorted[every_points * i:(every_points * (i + 1))])
                temp_var_median = np.median(var_x_sorted[every_points * i:(every_points * (i + 1))])
                temp_var_average = np.mean(var_x_sorted[every_points * i:(every_points * (i + 1))])
                x_mean.append(temp_x_mean ** 2)
                var_x_median.append(temp_var_median ** 2)
                var_x_average.append(temp_var_average ** 2)

            z1 = np.polyfit(x_mean, var_x_median, 1)
            p1 = np.poly1d(z1)

            self.x_mean = x_mean
            self.var_x_average = var_x_median
            self.memx_total = best_memx_total
            self.variation_x = variation_x

            sigma_relative = math.sqrt(abs(z1[0]) * math.pi / 2)
            sigma_absolute = math.sqrt(abs(z1[1]) * math.pi / 2)

            SSR = np.sum(np.square(var_x_median - (z1[0] * np.square(x_mean) + z1[1])))
            SST = np.sum(np.square(var_x_median - np.mean(var_x_median)))
            self.R_square = 1 - SSR / SST

        elif cluster_option == 'ew':
            group = 10
            segments = np.linspace(0, 1, group + 1)
            x_mean = np.zeros(group)
            variation_mean = np.zeros(group)
            for i in range(group):
                if i == group - 1:
                    mask = (x_sorted >= segments[i]) & (x_sorted <= segments[i + 1])
                else:
                    mask = (x_sorted >= segments[i]) & (x_sorted < segments[i + 1])

                x_mean[i] = np.mean(x_sorted[mask])
                variation_mean[i] = np.mean(var_x_sorted[mask])

            z1 = np.polyfit(np.square(x_mean), np.square(variation_mean), 1)
            p1 = np.poly1d(z1)

            self.x_mean = x_mean
            self.var_x_average = variation_mean
            self.memx_total = best_memx_total
            self.variation_x = variation_x

            sigma_relative = math.sqrt(abs(z1[0]) * math.pi / 2)
            sigma_absolute = math.sqrt(abs(z1[1]) * math.pi / 2)

            SSR = np.sum(np.square(np.square(variation_mean) - (z1[0] * np.square(x_mean) + z1[1])))
            SST = np.sum(np.square(np.square(variation_mean) - np.mean(np.square(variation_mean))))
            self.R_square = 1 - SSR / SST

        else:
            raise Exception('You have not specified a clustering method')

        return sigma_relative, sigma_absolute
