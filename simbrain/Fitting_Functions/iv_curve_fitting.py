import numpy as np
import pandas as pd
import torch


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


class IVCurve(object):
    # language=rst
    """
    Abstract base class for IV curve fitting of memristor.
    """
    
    def __init__(
            self,
            file,
            dictionary: dict = {},
            **kwargs,
    ):
        # language=rst
        """
        Abstract base class constructor.

        :param file: I-V curve data file.
        :param dictionary: Memristor device parameters.
        """
        # Read excel
        data = pd.DataFrame(pd.read_excel(
            file,
            sheet_name=0,
            header=None,
            names=[
                'Time(s)',
                'Excitation Voltage(V)',
                'Current Response(A)'
            ]
        ))

        # Initialize parameters
        self.alpha_off = 1
        self.alpha_on = 1

        # Read parameters
        self.v_off = dictionary['v_off']
        self.v_on = dictionary['v_on']

        self.G_off = dictionary['G_off']
        self.G_on = dictionary['G_on']

        self.k_off = dictionary['k_off']
        self.k_on = dictionary['k_on']
        self.P_off = dictionary['P_off']
        self.P_on = dictionary['P_on']

        # Read data
        self.voltage = np.array(data['Excitation Voltage(V)'])
        self.current = np.array(data['Current Response(A)'])

        if data['Time(s)'][:2].isnull().any():
            self.delta_t = 0.1
        else:
            self.delta_t = data['Time(s)'][1] - data['Time(s)'][0]

        if None in [self.G_off, self.G_on]:
            self.G_off = (1 + 0.12) * np.max(np.where(self.voltage > 0, self.current / self.voltage, 0))
            self.G_on = (1 - 0.12) * np.min(np.where(self.voltage < 0, self.current / self.voltage, 1))

        if None in [self.P_off, self.P_on]:
            self.P_off = 1
            self.P_on = 1


    def Memristor_conductance_model(
            self,
            alpha_off,
            alpha_on,
            x_init,
            V_write
    ):
        # language=rst
        """
        The Memristor model.

        :param alpha_off: Memristor parameter alpha off.
        :param alpha_on: Memristor parameter alpha on.
        :param x_init: The initial value of the internal state variable.
        :param V_write: The voltage of the input write pulse.
        """
        J1 = 1
        points = len(V_write)

        # initialization
        internal_state = [0 for i in range(points)]
        conductance_fit = [0 for i in range(points)]

        # conductance change
        internal_state[0] = x_init
        for i in range(points - 1):
            if V_write[i + 1] > self.v_off and V_write[i + 1] > 0:
                delta_x = self.k_off * ((V_write[i + 1] / self.v_off - 1) ** alpha_off) * J1 * (
                        (1 - internal_state[i]) ** self.P_off)
                internal_state[i + 1] = internal_state[i] + self.delta_t * delta_x

            elif V_write[i + 1] < 0 and V_write[i + 1] < self.v_on:
                delta_x = self.k_on * ((V_write[i + 1] / self.v_on - 1) ** alpha_on) * J1 * (
                        internal_state[i] ** self.P_on)
                internal_state[i + 1] = internal_state[i] + self.delta_t * delta_x

            else:
                delta_x = 0
                internal_state[i + 1] = internal_state[i]

            if internal_state[i + 1] < 0:
                internal_state[i + 1] = 0
            elif internal_state[i + 1] > 1:
                internal_state[i + 1] = 1

        # conductance calculation
        for i in range(points):
            conductance_fit[i] = self.G_off * internal_state[i] + self.G_on * (1 - internal_state[i])

        return internal_state, conductance_fit


    @timer
    def fitting(self, loss_option='rrmse_range'):
        # language=rst
        """
        Fit the I-V curve data using the baseline model.

        :param loss_option: The loss function to be used. Possible values are:
        - 'rmse': The root mean squared error.
        - 'rrmse_range': The relative root mean squared error using the difference between the maximum and minimum
        value to normalize.
        - 'rrmse_mean': The relative root mean squared error using the mean value to normalize.
        - 'rrmse_euclidean': The relative root mean squared error using the Euclidean norm to normalize.
        - 'rrmse_percent': The root mean squared relative error
        - 'rrmse_origin': The relative root mean relative squared error using the Euclidean norm to normalize.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # Set parameters' range
        J1 = 1
        alpha_off_nums = 10
        alpha_on_nums = 10
        alpha_off_list = torch.arange(alpha_off_nums) + 1
        alpha_on_list = torch.arange(alpha_on_nums) + 1
        alpha_off_list = alpha_off_list.to(device)
        alpha_on_list = alpha_on_list.to(device)

        if None in [self.k_off, self.k_on]:
            k_off_nums = 1000
            k_on_nums = 1000
            k_off_list = torch.logspace(-4, 9, k_off_nums, base=10)
            k_on_list = -torch.logspace(-4, 9, k_on_nums, base=10)
        else:
            k_off_nums = 1
            k_on_nums = 1
            k_off_list = torch.tensor(self.k_off)
            k_on_list = torch.tensor(self.k_on)
        k_off_list = k_off_list.to(device)
        k_on_list = k_on_list.to(device)

        zero_index = np.where(self.voltage == 0)
        self.voltage = np.delete(self.voltage, zero_index)
        self.current = np.delete(self.current, zero_index)

        V_write = self.voltage
        if V_write[0] > 0:
            start_point_r = 0
            points_r = np.sum(V_write > 0)
            start_point_d = start_point_r + points_r
            points_d = np.sum(V_write < 0)
        else:
            start_point_d = 0
            points_d = np.sum(V_write < 0)
            start_point_r = start_point_d + points_d
            points_r = np.sum(V_write > 0)

        V_write_r = torch.tensor(V_write[start_point_r: start_point_r + points_r])
        V_write_d = torch.tensor(V_write[start_point_d: start_point_d + points_d])
        V_write_r = V_write_r.to(device)
        V_write_d = V_write_d.to(device)
        current_r = torch.tensor(self.current[start_point_r: start_point_r + points_r])
        current_d = torch.tensor(self.current[start_point_d: start_point_d + points_d])

        x_init_r = (current_r[0] / self.voltage[0] - self.G_on) / (self.G_off - self.G_on)
        x_init_r = x_init_r if x_init_r > 0 else 0
        x_init_r = x_init_r if x_init_r < 1 else 1
        x_init_d = (current_d[0] / self.voltage[points_r] - self.G_on) / (self.G_off - self.G_on)
        x_init_d = x_init_d if x_init_d > 0 else 0
        x_init_d = x_init_d if x_init_d < 1 else 1

        # positive
        mem_x_r = torch.zeros([points_r, alpha_off_nums, k_off_nums])
        mem_x_r = mem_x_r.to(device)
        mem_x_r[0] = x_init_r * torch.ones([alpha_off_nums, k_off_nums])
        for j in range(points_r - 1):
            mem_x_r[j + 1] = torch.where(
                V_write_r[j + 1] > self.v_off and V_write_r[j + 1] > 0,
                k_off_list.expand(alpha_off_nums, k_off_nums)
                * ((V_write_r[j + 1] / self.v_off - 1) ** alpha_off_list.expand(k_off_nums, alpha_off_nums).T)
                * J1
                * (1 - mem_x_r[j]) ** self.P_off
                * self.delta_t
                + mem_x_r[j],
                mem_x_r[j]
            )
            mem_x_r[j + 1] = torch.where(mem_x_r[j + 1] < 0, 0, mem_x_r[j + 1])
            mem_x_r[j + 1] = torch.where(mem_x_r[j + 1] > 1, 1, mem_x_r[j + 1])

        mem_x_r_T = mem_x_r.permute(1, 2, 0)
        mem_c_r = self.G_off * mem_x_r_T + self.G_on * (1 - mem_x_r_T)
        current_fit_r = mem_c_r * V_write_r
        current_r = current_r.to(device)

        if loss_option == 'rmse':
            i_diff = current_fit_r - current_r
            INDICATOR_r = torch.sqrt(torch.sum(i_diff * i_diff, dim=2) / points_r)
        elif loss_option == 'rrmse_range':
            i_diff = current_fit_r - current_r
            INDICATOR_r = torch.sqrt(torch.sum(i_diff * i_diff, dim=2) / points_r) / (
                        torch.max(current_r) - torch.min(current_r))
        elif loss_option == 'rrmse_mean':
            i_diff = current_fit_r - current_r
            INDICATOR_r = torch.sqrt(torch.sum(i_diff * i_diff, dim=2) / points_r) / torch.mean(current_r)
        elif loss_option == 'rrmse_euclidean':
            i_diff = current_fit_r - current_r
            INDICATOR_r = torch.sqrt(torch.sum(i_diff * i_diff, dim=2) / torch.dot(current_r, current_r) / points_r)
        elif loss_option == 'rrmse_percent':
            i_diff_percent = (current_fit_r - current_r) / current_r
            INDICATOR_r = torch.sqrt(torch.sum(i_diff_percent * i_diff_percent, dim=2) / points_r)
        elif loss_option == 'rrmse_origin':
            i_diff_percent = (current_fit_r - current_r) / current_r
            INDICATOR_r = torch.sqrt(
                torch.sum(i_diff_percent * i_diff_percent, dim=2) / torch.dot(current_r, current_r) / points_r)

        # negative
        mem_x_d = torch.zeros([points_d, alpha_on_nums, k_on_nums])
        mem_x_d = mem_x_d.to(device)
        mem_x_d[0] = x_init_d * torch.ones([alpha_on_nums, k_on_nums])
        for j in range(points_d - 1):
            mem_x_d[j + 1] = torch.where(
                V_write_d[j + 1] < 0 and V_write_d[j + 1] < self.v_on,
                k_on_list.expand(alpha_on_nums, k_on_nums)
                * ((V_write_d[j + 1] / self.v_on - 1) ** alpha_on_list.expand(k_on_nums, alpha_on_nums).T)
                * J1
                * (1 - mem_x_d[j]) ** self.P_on
                * self.delta_t
                + mem_x_d[j],
                mem_x_d[j],
            )
            mem_x_d[j + 1] = torch.where(mem_x_d[j + 1] < 0, 0, mem_x_d[j + 1])
            mem_x_d[j + 1] = torch.where(mem_x_d[j + 1] > 1, 1, mem_x_d[j + 1])

        mem_x_d_T = mem_x_d.permute(1, 2, 0)
        mem_c_d = self.G_off * mem_x_d_T + self.G_on * (1 - mem_x_d_T)
        current_fit_d = mem_c_d * V_write_d
        current_d = current_d.to(device)
        # RRMSE calculation
        if loss_option == 'rmse':
            i_diff = current_fit_d - current_d
            INDICATOR_d = torch.sqrt(torch.sum(i_diff * i_diff, dim=2) / points_d)
        elif loss_option == 'rrmse_range':
            i_diff = current_fit_d - current_d
            INDICATOR_d = torch.sqrt(torch.sum(i_diff * i_diff, dim=2) / points_d) / (
                        torch.max(current_d) - torch.min(current_d))
        elif loss_option == 'rrmse_mean':
            i_diff = current_fit_d - current_d
            INDICATOR_d = torch.sqrt(torch.sum(i_diff * i_diff, dim=2) / points_d) / torch.mean(torch.abs(current_d))
        elif loss_option == 'rrmse_euclidean':
            i_diff = (current_fit_d - current_d)
            INDICATOR_d = torch.sqrt(torch.sum(i_diff * i_diff, dim=2) / torch.dot(current_d, current_d) / points_d)
        elif loss_option == 'rrmse_percent':
            i_diff_percent = (current_fit_d - current_d) / current_d
            INDICATOR_d = torch.sqrt(torch.sum(i_diff_percent * i_diff_percent, dim=2) / points_d)
        elif loss_option == 'rrmse_origin':
            i_diff_percent = (current_fit_d - current_d) / current_d
            INDICATOR_d = torch.sqrt(
                torch.sum(i_diff_percent * i_diff_percent, dim=2) / torch.dot(current_d, current_d) / points_d)

        self.loss_r = torch.min(INDICATOR_r)
        self.loss_d = torch.min(INDICATOR_d)
        min_x = torch.argmin(INDICATOR_r)
        min_y = torch.argmin(INDICATOR_d)
        self.alpha_off = alpha_off_list[min_x // k_off_nums].item()
        self.alpha_on = alpha_on_list[min_y // k_on_nums].item()
        self.k_off = k_off_list[min_x % k_off_nums].item()
        self.k_on = k_on_list[min_y % k_on_nums].item()

        torch.cuda.empty_cache()

        return self.alpha_off, self.alpha_on
