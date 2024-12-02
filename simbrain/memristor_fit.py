import os
import json
import copy
import shutil
import numpy as np
import pandas as pd
import pickle
from simbrain.Fitting_Functions.iv_curve_fitting import IVCurve
from simbrain.Fitting_Functions.conductance_fitting import Conductance
from simbrain.Fitting_Functions.variation_fitting import Variation
from simbrain.Fitting_Functions.retention_loss_fitting import RetentionLoss
from simbrain.Fitting_Functions.aging_effect_fitting import AgingEffect
from simbrain.Fitting_Functions.stuck_at_fault_fitting import StuckAtFault


class MemristorFitting(object):
    # language=rst
    """
    Abstract base class for memristor fitting.
    """

    def __init__(
            self,
            sim_params: dict = {},
            my_memristor: dict = {},
            **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.
        :param sim_params: Memristor device to be used in learning.
        :param my_memristor: The parameters of the memristor device.
        """
        self.device_name = sim_params['device_name']
        self.c2c_variation = sim_params['c2c_variation']
        self.d2d_variation = sim_params['d2d_variation']
        self.stuck_at_fault = sim_params['stuck_at_fault']
        self.retention_loss = sim_params['retention_loss']
        self.aging_effect = sim_params['aging_effect']
        self.wire_width = sim_params['wire_width']
        self.mem_size = my_memristor['mem_size']
        self.relax_ratio_row = my_memristor['relax_ratio_row']
        self.relax_ratio_col = my_memristor['relax_ratio_col']        
        self.fitting_record = my_memristor


    def copy_data(self, root, ref, obj, data_file):
        # language=rst
        """
        Copy data from the reference directory to the object directory.

        :param root: Root directory of the data files.
        :param ref: Reference directory to store all data files.
        :param obj: Object directory to store the data files to be used.
        :param data_file: Data filename.
        """
        shutil.copy(
            root + ref + data_file,
            root + obj + data_file,
        )


    def delete_data(self, root, obj):
        # language=rst
        """
        Delete useless data in the object directory.

        :param root: Root directory of the data files.
        :param obj: Object directory of the data files.
        """
        path = root + obj
        for file in os.listdir(path):
            if file.endswith('.xlsx'):
                os.remove(path + '/' + file)


    def check_required_data(self, v_off, v_on, delta_t, duty_ratio):
        # language=rst
        """
        Check if required data is present.

        :param v_off: Threshold voltage for potentiation.
        :param v_on: Threshold voltage for depression.
        :param delta_t: Period of input pulse.
        :param duty_ratio: Duty cycle of input pulse.
        """
        if None in [self.mem_size, self.relax_ratio_col, self.relax_ratio_row]:
            raise Exception("Error! Missing mem_size data!")
        if None in [v_on, v_off]:
            raise Exception("Error! Missing v_on/v_off data!")
        if None in [delta_t, duty_ratio]:
            raise Exception("Error! Missing pulse time data!")


    def mem_fitting(self):
        # language=rst
        """
        Use memristor device model to fit memristor raw data.
        """
        # %% Obtain memristor parameters
        mem_info = copy.copy(self.fitting_record)

        k_off = mem_info['k_off']
        k_on = mem_info['k_on']
        v_off = mem_info['v_off']
        v_on = mem_info['v_on']
        alpha_off = mem_info['alpha_off']
        alpha_on = mem_info['alpha_on']
        P_off = mem_info['P_off']
        P_on = mem_info['P_on']
        G_off = mem_info['G_off']
        G_on = mem_info['G_on']
        v_write_lut = mem_info['v_write_lut']
        v_read = mem_info['v_read']
        delta_t = mem_info['delta_t']
        duty_ratio = mem_info['duty_ratio']

        sigma_relative = mem_info['sigma_relative']
        sigma_absolute = mem_info['sigma_absolute']

        Goff_sigma = mem_info['Goff_sigma']
        Gon_sigma = mem_info['Gon_sigma']
        Poff_sigma = mem_info['Poff_sigma']
        Pon_sigma = mem_info['Pon_sigma']

        SAF_lambda = mem_info['SAF_lambda']
        SAF_ratio = mem_info['SAF_ratio']
        SAF_delta = mem_info['SAF_delta']

        retention_loss_tau_reciprocal = mem_info['retention_loss_tau_reciprocal']
        retention_loss_beta = mem_info['retention_loss_beta']

        Aging_off = mem_info['Aging_off']
        Aging_on = mem_info['Aging_on']

        # Data paths
        root = os.path.dirname(os.path.dirname(__file__))
        ref = "/reference_memristor_data"
        obj = "/memristor_data"

        G_th_path = "/G_variation.xlsx"
        iv_curve_path = "/iv_c.xlsx"
        conductance_path = "/conductance_deletehead.xlsx"
        retention_loss_path = "/retention_loss.xlsx"
        aging_effect_path = "/aging_effect.xlsx"
        saf_path = "/saf_data.xlsx"

        self.delete_data(root, obj)
        self.check_required_data(v_off, v_on, delta_t, duty_ratio)

        # Fitting process
        print("Start Memristor Fitting:\n")

        # %% Pre-deployment SAF
        if self.stuck_at_fault:
            if None not in [SAF_lambda, SAF_ratio]:
                pass
            elif not os.path.isfile(root + ref + saf_path):
                raise Exception("Error! Missing data files.\nFailed to update SAF_lambda, SAF_ratio.")
            else:
                print("Pre-deployment Stuck at Fault calculating...")
                self.copy_data(root, ref, obj, saf_path)
                SAF_lambda, SAF_ratio = StuckAtFault(root + obj + saf_path).pre_deployment_fitting()
                mem_info.update(
                    {
                        "SAF_lambda": SAF_lambda,
                        "SAF_ratio": SAF_ratio
                    }
                )

        # %% G_off, G_on
        if os.path.isfile(root + ref + G_th_path):
            self.copy_data(root, ref, obj, G_th_path)
            data = pd.DataFrame(pd.read_excel(
                root + obj + G_th_path,
                sheet_name='Sheet1',
                header=None,
                index_col=None
            ))
            data.columns = ['G_off', 'G_on']

            device_nums = data.shape[0]
            G_off_variation = np.array(data['G_off'])
            G_on_variation = np.array(data['G_on'])
        elif os.path.isfile(root + ref + conductance_path):
            self.copy_data(root, ref, obj, conductance_path)
            data = pd.DataFrame(pd.read_excel(
                root + obj + conductance_path,
                sheet_name='Sheet1',
                header=None,
                index_col=None
            ))
            data.columns = ['Pulse Voltage(V)', 'Read Voltage(V)'] + list(data.columns[2:] - 2)

            V_write = np.array(data['Pulse Voltage(V)'])
            points_r = np.sum(V_write > 0)
            points_d = np.sum(V_write < 0)
            read_voltage = np.array(data['Read Voltage(V)'])[0]

            device_nums = data.shape[1] - 2
            G_off_variation = np.zeros(device_nums)
            G_on_variation = np.zeros(device_nums)
            G_on_num = int(points_d / 20)  + 1
            G_off_num = int(points_r / 20) + 1
        
            for i in range(device_nums):
                G_off_variation[i] = np.average(
                    data[i][points_r - G_off_num:points_r] / read_voltage
                )
                G_on_variation[i] = np.average(
                    data[i][points_r + points_d - G_on_num:] / read_voltage
                ) 
        else:
            raise Exception("Error! Missing data files.\nFailed to update G_off, G_on.")

        if None in [G_off, G_on]:
            G_off = np.mean(G_off_variation)
            G_on = np.mean(G_on_variation)

        mem_info.update(
            {
                "G_off": G_off,
                "G_on": G_on
            }
        )

        P_off_variation = np.zeros(device_nums)
        P_on_variation = np.zeros(device_nums)

        # %% D2D variation G_off/G_on
        if self.d2d_variation in [1, 2]:
            if None not in [Gon_sigma, Goff_sigma]:
                pass
            elif not os.path.isfile(root + ref + conductance_path):
                raise Exception("Error! Missing data files.\nFailed to update Goff_sigma, Gon_sigma.")
            else:
                print("Device to Device Variation calculating...")
                self.copy_data(root, ref, obj, conductance_path)
                Goff_mu, Goff_sigma, Gon_mu, Gon_sigma = Variation(
                    root + obj + conductance_path,
                    G_off_variation,
                    G_on_variation,
                    P_off_variation,
                    P_on_variation,
                    mem_info
                ).d2d_G_fitting()
                mem_info.update(
                    {
                        "Goff_sigma": Goff_sigma,
                        "Gon_sigma": Gon_sigma,
                    }
                )

        # %% Baseline Model(IV curve)
        print("Baseline Model calculating...")
        if None not in [alpha_off, alpha_on]:
            pass
        elif not os.path.isfile(root + ref + iv_curve_path):
            print("Warning! Missing data files.\nDefault alpha is 5.")
            mem_info.update(
                {
                    "alpha_off": 5,
                    "alpha_on": 5
                }
            )
        else:
            self.copy_data(root, ref, obj, iv_curve_path)
            G_off_tmp = 6.3315e-3
            G_on_tmp = 1.3088e-3
            mem_info.update(
                {             
                    "G_off": G_off_tmp,
                    "G_on": G_on_tmp
                }
            )
            alpha_off, alpha_on = IVCurve(root + obj + iv_curve_path, mem_info).fitting(loss_option='rrmse_percent')
            mem_info.update(
                {
                    "alpha_off": alpha_off,
                    "alpha_on": alpha_on,
                    "G_off": G_off,
                    "G_on": G_on
                }
            )

        # %% Baseline Model(Conductance)
        if None not in [P_off, P_on, k_off, k_on, v_write_lut, v_read]:
            pass
        elif not os.path.isfile(root + ref + conductance_path):
            raise Exception("Error! Missing data files.\nFailed to update P_off, P_on, k_off, k_on.")
        else:
            self.copy_data(root, ref, obj, conductance_path)
            conductance_temp = Conductance(root + obj + conductance_path, mem_info)
            P_off, P_on, k_off, k_on, v_write_lut, v_read = conductance_temp.fitting(loss_option='rmse')
            mem_info.update(
                {
                    "P_off": P_off,
                    "P_on": P_on,
                    "k_off": k_off,
                    "k_on": k_on,
                    "v_write_lut": v_write_lut.item(),
                    "v_read": v_read.item()
                }
            )

        V_write_lut = v_write_lut

        if self.d2d_variation in [1, 3] or self.c2c_variation:
            try:
                self.copy_data(root, ref, obj, conductance_path)
                conductance_temp = Conductance(root + obj + conductance_path, mem_info)
                P_off_variation, P_on_variation = conductance_temp.mult_P_fitting(G_off_variation, G_on_variation)
            except:
                raise Exception("Error! Failed to calculate the Variation.")

        # %% D2D variation nonlinearity
        if self.d2d_variation in [1, 3]:
            if None not in [Pon_sigma, Poff_sigma]:
                pass
            elif not os.path.isfile(root + ref + conductance_path):
                raise Exception("Error! Missing data files.\nFailed to update Poff_sigma, Pon_sigma.")
            else:
                print("Device to Device Variation(Non-linearity) calculating...")
                if os.path.isfile(root + obj + conductance_path):
                    P_off_variation, P_on_variation = conductance_temp.mult_P_fitting(G_off_variation, G_on_variation,
                                                                                      loss_option='rmse')
                else:
                    self.copy_data(root, ref, obj, conductance_path)
                    conductance_temp = Conductance(root + obj + conductance_path, mem_info)
                    P_off_variation, P_on_variation = conductance_temp.mult_P_fitting(G_off_variation, G_on_variation,
                                                                                      loss_option='rmse')
                variation_temp = Variation(
                    root + obj + conductance_path,
                    G_off_variation,
                    G_on_variation,
                    P_off_variation,
                    P_on_variation,
                    mem_info
                )
                _, Poff_sigma, _, Pon_sigma = variation_temp.d2d_P_fitting()
                mem_info.update(
                    {
                        "Poff_sigma": Poff_sigma,
                        "Pon_sigma": Pon_sigma
                    }
                )

        # %% C2C variation
        if self.c2c_variation:
            if None not in [sigma_relative, sigma_absolute]:
                pass
            elif not os.path.isfile(root + ref + conductance_path):
                raise Exception("Error! Missing data files.\nFailed to update sigma_relative, sigma_absolute.")
            else:
                print("Cycle to Cycle Variation calculating...")
                try:
                    sigma_relative, sigma_absolute = variation_temp.c2c_fitting(cluster_option='ew')
                except:
                    if os.path.isfile(root + obj + conductance_path):
                        P_off_variation, P_on_variation = conductance_temp.mult_P_fitting(G_off_variation,
                                                                                          G_on_variation,
                                                                                          loss_option='rmse')
                    else:
                        self.copy_data(root, ref, obj, conductance_path)
                        conductance_temp = Conductance(root + obj + conductance_path, mem_info)
                        P_off_variation, P_on_variation = conductance_temp.mult_P_fitting(G_off_variation,
                                                                                          G_on_variation,
                                                                                          loss_option='rmse')
                    variation_temp = Variation(
                        root + obj + conductance_path,
                        G_off_variation,
                        G_on_variation,
                        P_off_variation,
                        P_on_variation,
                        mem_info
                    )
                    sigma_relative, sigma_absolute = variation_temp.c2c_fitting(cluster_option='ew')
                mem_info.update(
                    {
                        "sigma_relative": sigma_relative,
                        "sigma_absolute": sigma_absolute
                    }
                )

        # %% Post-deployment SAF
        if self.stuck_at_fault:
            if None not in [SAF_delta]:
                pass
            elif not os.path.isfile(root + ref + saf_path):
                raise Exception("Error! Missing data files.\nFailed to update SAF_delta.")
            else:
                print("Post-deployment Stuck at Fault calculating...")
                SAF_delta = StuckAtFault(root + obj + saf_path).post_deployment_fitting()
                mem_info.update(
                    {
                        "SAF_delta": SAF_delta
                    }
                )

        # %% Retention loss
        if self.retention_loss:
            if None not in [retention_loss_tau_reciprocal, retention_loss_beta]:
                pass
            elif not os.path.isfile(root + ref + retention_loss_path):
                raise Exception("Error! Missing data files.\nFailed to update retention_loss_tau, retention_loss_beta.")
            else:
                print("Retention Loss calculating...")
                self.copy_data(root, ref, obj, retention_loss_path)
                retention_loss_tau_reciprocal, retention_loss_beta = RetentionLoss(root + obj + retention_loss_path).fitting()
                mem_info.update(
                    {
                        "retention_loss_tau_reciprocal": retention_loss_tau_reciprocal,
                        "retention_loss_beta": retention_loss_beta
                    }
                )

        # %% Aging effect
        if self.aging_effect in [1, 2]:
            if None not in [Aging_off, Aging_on]:
                pass
            elif not os.path.isfile(root + ref + aging_effect_path):
                raise Exception("Error! Missing data files.\nFailed to update Aging_off, Aging_on.")
            else:
                print("Aging Effect calculating...")
                self.copy_data(root, ref, obj, aging_effect_path)
                aging_cal = AgingEffect(root + obj + aging_effect_path, mem_info)
                if self.aging_effect == 1:
                    Aging_off, Aging_on = aging_cal.fitting_equation1()
                else:
                    Aging_off, Aging_on = aging_cal.fitting_equation2()
                mem_info.update(
                    {
                        "Aging_off": Aging_off,
                        "Aging_on": Aging_on
                    }
                )

        print("\nEnd Memristor Fitting.")

        self.mem_info_update(mem_info)
        self.mem_lut_update(mem_info, V_write_lut)

        return self.fitting_record


    def mem_info_update(self, mem_info):
        # language=rst
        """
        Update parameters of the memristor device in json files.

        :param mem_info: The parameters of the memristor device.
        """
        self.fitting_record = mem_info
        json.dumps(self.fitting_record, indent=4, separators=(',', ':'))
        with open('../../memristor_device_info.json', 'r') as f:
            memristor_info_dict = json.load(f)
        memristor_info_dict['mine'] = self.fitting_record
        with open('../../memristor_device_info.json', 'w') as f:
            json.dump(memristor_info_dict, f, indent=2)


    def mem_lut_update(self, mem_info, V_write_lut):
        # language=rst
        """
        Update parameters of the memristor device in the weight-to-voltage look-up-table.

        :param mem_info: The parameters of the memristor device.
        :param V_write_lut: The voltage of the input write pulse.
        """
        v_off = mem_info['v_off']
        v_on = mem_info['v_on']
        rise_ending = 0.97
        setting_step = 1.2
        min_states_num = 50
        states_step = 10
        max_states_num = 1001
        max_V_reset = 0

        if V_write_lut > v_off and V_write_lut < 2 * v_off:
            V_write_lut = np.full(max_states_num, V_write_lut)
        elif V_write_lut < v_off:
            raise Exception("V_write given is smaller than threshold voltage!")
        else:
            V_write_lut = np.full(max_states_num, 2 * v_off)
            print("[Warning] V_write given is bigger than twice threshold voltage!")

        for states_num in range(min_states_num, max_states_num, states_step):
            lut_state, lut_conductance = self.lut_state_generate(V_write_lut, mem_info, 0)
            if lut_state[states_num] > rise_ending:
                best_states_num = states_num
                break
            if states_num == max_states_num - 1:
                best_states_num = states_num
                print("[Warning] conductance cannot close to Goff!")

        for init_state in range(10, 0, -1):
            V_reset = [0, v_on]
            init_state *= 0.1
            while True:
                reset_state, _ = self.lut_state_generate(V_reset, mem_info, init_state)
                if reset_state[1] == 0:
                    break
                else:
                    V_reset[1] = V_reset[1] * setting_step
            if np.abs(V_reset[1]) > np.abs(max_V_reset):
                max_V_reset = V_reset[1]

        mine_lut = {
            'total_no': best_states_num,
            'voltage': V_write_lut[0],
            'cycle:': mem_info['delta_t'],
            'duty ratio': mem_info['duty_ratio'],
            'V_reset': max_V_reset,
            'conductance': lut_conductance[0:best_states_num + 1]
        }

        with open('../../memristor_lut.pkl', 'rb') as f:
            mem_lut = pickle.load(f)
        mem_lut['mine'] = mine_lut
        with open('../../memristor_lut.pkl', 'wb') as f:
            pickle.dump(mem_lut, f)


    def lut_state_generate(self, V_write, mem_info, x_init):
        # language=rst
        """
        Generate a weight-to-voltage look-up-table to store the parameters of the memristor device.

        :param V_write: The voltage of the input write pulse.
        :param mem_info: The parameters of the memristor device.
        :param x_init: The initial value of the internal state variable.
        """
        J1 = 1
        points = len(V_write)

        # initialization
        internal_state = [0 for i in range(points)]
        conductance_fit = [0 for i in range(points)]

        # conductance change
        internal_state[0] = x_init
        for i in range(points - 1):
            if V_write[i + 1] > mem_info['v_off'] and V_write[i + 1] > 0:
                delta_x = mem_info['k_off'] * (
                        (V_write[i + 1] / mem_info['v_off'] - 1) ** mem_info['alpha_off']) * J1 * (
                                  (1 - internal_state[i]) ** mem_info['P_off'])
                internal_state[i + 1] = internal_state[i] + mem_info['delta_t'] * mem_info['duty_ratio'] * delta_x
            elif V_write[i + 1] < mem_info['v_on'] and V_write[i + 1] < 0:
                delta_x = mem_info['k_on'] * ((V_write[i + 1] / mem_info['v_on'] - 1) ** mem_info['alpha_on']) * J1 * (
                        internal_state[i] ** mem_info['P_on'])
                internal_state[i + 1] = internal_state[i] + mem_info['delta_t'] * mem_info['duty_ratio'] * delta_x
            else:
                delta_x = 0
                internal_state[i + 1] = internal_state[i]
            if internal_state[i + 1] < 0:
                internal_state[i + 1] = 0
            elif internal_state[i + 1] > 1:
                internal_state[i + 1] = 1

        # conductance calculation
        for i in range(points):
            conductance_fit[i] = mem_info['G_off'] * internal_state[i] + mem_info['G_on'] * (1 - internal_state[i])

        return internal_state, conductance_fit
