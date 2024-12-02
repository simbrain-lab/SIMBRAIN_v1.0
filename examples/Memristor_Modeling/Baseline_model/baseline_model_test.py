import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append('../../../')
from simbrain.Fitting_Functions.iv_curve_fitting import IVCurve
from simbrain.Fitting_Functions.conductance_fitting import Conductance


def main():
    # Fit
    file_G = "../../../reference_memristor_data/G_variation_CMS.xlsx"
    file_iv = "../../../reference_memristor_data/iv_curve_CMS.xlsx"
    file_c = "../../../reference_memristor_data/conductance_CMS.xlsx"

    with open("../../../memristor_data/my_memristor.json") as f:
        dict = json.load(f)
    dict.update(
        {
            'v_off': 0.2,
            'v_on': -0.2,
            'G_off': None,
            'G_on': None,
            'alpha_off': None,
            'alpha_on': None,
            'k_off': None,
            'k_on': None,
            'P_off': None,
            'P_on': None,
            'delta_t': 260e-9,
            "duty_ratio": 0.6154
        }
    )
    if os.path.isfile(file_G):
        data = pd.DataFrame(pd.read_excel(
            file_G,
            sheet_name='Sheet1',
            header=None,
            index_col=None
        ))
        data.columns = ['G_off', 'G_on']
        G_off_list = np.array(data['G_off'])
        G_on_list = np.array(data['G_on'])
        G_off = np.mean(G_off_list)
        G_on = np.mean(G_on_list)
    else:
        data = pd.DataFrame(pd.read_excel(
            file_c,
            sheet_name=0,
            header=None,
            index_col=None,
        ))
        data.columns = ['Pulse Voltage(V)', 'Read Voltage(V)'] + list(data.columns[2:] - 2)

        V_write = np.array(data['Pulse Voltage(V)'])
        points_r = np.sum(V_write > 0)
        points_d = np.sum(V_write < 0)
        read_voltage = np.array(data['Read Voltage(V)'])[0]

        device_nums = data.shape[1] - 2
        G_off_list = np.zeros(device_nums)
        G_on_list = np.zeros(device_nums)
        G_on_num = int(points_d / 20)  + 1
        G_off_num = int(points_r / 20) + 1
    
        for i in range(device_nums):
            G_off_list[i] = np.average(
                data[i][points_r - G_off_num:points_r] / read_voltage
            )
            G_on_list[i] = np.average(
                data[i][points_r + points_d - G_on_num:] / read_voltage
            ) 

        G_off = np.mean(G_off_list)
        G_on = np.mean(G_on_list)
    
    G_off_temp = 6.3315e-3
    G_on_temp = 1.3088e-3
    dict.update(
        {
            'v_off': 0.2,
            'v_on': -0.2,
            'G_off': G_off_temp,
            'G_on': G_on_temp,
        }
    )

    if os.path.isfile(file_iv):
        print('Going through IV curve fitting process')
        # loss_list = ['rmse', 'rrmse_range', 'rrmse_mean', 'rrmse_euclidean', 'rrmse_percent']
        loss_list = ['rrmse_percent']
        for loss in loss_list:
            exp_1 = IVCurve(file_iv, dict)
            alpha_off, alpha_on = exp_1.fitting(loss_option=loss)
            df = pd.DataFrame(
                {
                    'value': [exp_1.G_off, exp_1.G_on, alpha_off, alpha_on]
                },
                index=['G_off_iv', 'G_on_iv', 'alpha_off', 'alpha_on']
            )
            print(df)
    else:
        print('No IV curve data!')
        alpha_off, alpha_on = 5, 5  # TODO: change default setting

    dict.update(
        {
            'G_off': G_off,
            'G_on': G_on,
            "alpha_off": alpha_off,
            "alpha_on": alpha_on,
            'v_off': 0.2,
            'v_on': -0.2,
        }
    )

    if os.path.isfile(file_c):
        # loss_list = ['rmse', 'rrmse_range', 'rrmse_mean', 'rrmse_euclidean', 'rrmse_percent']
        loss_list = ['rmse']
        for loss in loss_list:
            print('Loss option:{}'.format(loss))
            exp_2 = Conductance(file_c, dict)
            P_off, P_on, k_off, k_on, _, _ = exp_2.fitting(loss_option=loss)
            dict.update(
                {
                    "P_off": P_off,
                    "P_on": P_on,
                    "k_off": k_off,
                    "k_on": k_on
                }
            )

            # Output
            df = pd.DataFrame(
                {
                    'value': [P_off, P_on, k_off, k_on]
                },
                index=['P_off', 'P_on', 'k_off', 'k_on']
            )
            print(df)
            with open("fitting_record.json", "w") as f:
                json.dump(dict, f, indent=2)

            # Output conductance scatter
            current_r = np.array(exp_2.data[:])[exp_2.start_point_r: exp_2.start_point_r + exp_2.points_r, 2:]
            current_d = np.array(exp_2.data[:])[exp_2.start_point_d: exp_2.start_point_d + exp_2.points_d, 2:]
            conductance_r = current_r / exp_2.read_voltage
            conductance_d = current_d / exp_2.read_voltage
            V_write_r = exp_2.V_write[exp_2.start_point_r: exp_2.start_point_r + exp_2.points_r]
            V_write_d = exp_2.V_write[exp_2.start_point_d: exp_2.start_point_d + exp_2.points_d]
            x_r = (conductance_r - G_on_list[np.newaxis, :]) / (G_off_list[np.newaxis, :] - G_on_list[np.newaxis, :])
            x_d = (conductance_d - G_on_list[np.newaxis, :]) / (G_off_list[np.newaxis, :] - G_on_list[np.newaxis, :])
            x_scatter = pd.DataFrame(np.concatenate((np.array(x_r), np.array(x_d))))
            # x_scatter.to_excel("../Fitting_results/x_scatter.xlsx", index=False, header=False)

            # Output conductance curve
            J1 = 1
            loss = 'rmse'
            P_off_list, P_on_list = exp_2.mult_P_fitting(G_off_list, G_on_list, loss_option=loss)
            x_init_r = np.maximum(np.zeros_like(np.array(x_r)[0]), np.array(x_r)[0])
            x_init_d = np.minimum(np.ones_like(np.array(x_d)[0]), np.array(x_d)[0])
            mem_x_r = np.zeros([exp_2.points_r, exp_2.device_nums])
            mem_x_d = np.zeros([exp_2.points_d, exp_2.device_nums])
            mem_x_r[0] = x_init_r
            mem_x_d[0] = x_init_d

            for i in range(exp_2.points_r - 1):
                for j in range(exp_2.device_nums):
                    mem_x_r[i + 1, j] = (
                            exp_2.k_off
                            * ((V_write_r[i + 1] / exp_2.v_off - 1) ** exp_2.alpha_off)
                            * J1
                            * (1 - mem_x_r[i, j]) ** P_off_list[j]
                            * exp_2.delta_t
                            * exp_2.duty_ratio
                            + mem_x_r[i, j]
                    )
                    mem_x_r[i + 1, j] = np.where(mem_x_r[i + 1, j] < 0, 0, mem_x_r[i + 1, j])
                    mem_x_r[i + 1, j] = np.where(mem_x_r[i + 1, j] > 1, 1, mem_x_r[i + 1, j])
            for i in range(exp_2.points_d - 1):
                for j in range(exp_2.device_nums):
                    mem_x_d[i + 1, j] = (
                            exp_2.k_on
                            * ((V_write_d[i + 1] / exp_2.v_on - 1) ** exp_2.alpha_on)
                            * J1
                            * mem_x_d[i, j] ** P_on_list[j]
                            * exp_2.delta_t
                            * exp_2.duty_ratio
                            + mem_x_d[i, j]
                    )
                    mem_x_d[i + 1, j] = np.where(mem_x_d[i + 1, j] < 0, 0, mem_x_d[i + 1, j])
                    mem_x_d[i + 1, j] = np.where(mem_x_d[i + 1, j] > 1, 1, mem_x_d[i + 1, j])

            data_curve = pd.DataFrame(np.concatenate((np.array(mem_x_r), np.array(mem_x_d))))
            # data_curve.to_excel("../Fitting_results/conductance_curve.xlsx", index=False, header=False)

            x_r = (conductance_r - exp_2.G_on) / (exp_2.G_off - exp_2.G_on)
            x_d = (conductance_d - exp_2.G_on) / (exp_2.G_off - exp_2.G_on)
            x_init_r = x_r[0]
            x_init_d = x_d[0] #1
            mem_x_r = np.zeros(exp_2.points_r)
            mem_x_d = np.zeros(exp_2.points_d)
            mem_x_r[0] = np.average(x_init_r)
            mem_x_d[0] = np.average(x_init_d)

            for i in range(exp_2.points_r - 1):
                mem_x_r[i + 1] = (
                        k_off
                        * ((V_write_r[i + 1] / exp_2.v_off - 1) ** exp_2.alpha_off)
                        * J1
                        * (1 - mem_x_r[i]) ** P_off
                        * exp_2.delta_t
                        * exp_2.duty_ratio
                        + mem_x_r[i]
                )
                mem_x_r[i + 1] = np.where(mem_x_r[i + 1] < 0, 0, mem_x_r[i + 1])
                mem_x_r[i + 1] = np.where(mem_x_r[i + 1] > 1, 1, mem_x_r[i + 1])
            for i in range(exp_2.points_d - 1):
                mem_x_d[i + 1] = (
                        k_on
                        * ((V_write_d[i + 1] / exp_2.v_on - 1) ** exp_2.alpha_on)
                        * J1
                        * mem_x_d[i] ** P_on
                        * exp_2.delta_t
                        * exp_2.duty_ratio
                        + mem_x_d[i]
                )
                mem_x_d[i + 1] = np.where(mem_x_d[i + 1] < 0, 0, mem_x_d[i + 1])
                mem_x_d[i + 1] = np.where(mem_x_d[i + 1] > 1, 1, mem_x_d[i + 1])
            memx_total = np.concatenate((mem_x_r, mem_x_d))
            x_total = np.concatenate((x_r, x_d))

            # Output best curve
            best_curve = pd.DataFrame(memx_total)
            # best_curve.to_excel("../Fitting_results/best_curve.xlsx", index=False, header=False)

            # Output error
            x_r = (conductance_r - exp_2.G_on) / (G_off - exp_2.G_on)
            x_d = (conductance_d - exp_2.G_on) / (G_off - exp_2.G_on)
            x = np.concatenate((x_r, x_d))
            if loss == 'rmse':
                df_e = pd.DataFrame(
                    {
                        'value': [
                            np.min(exp_2.loss_r.numpy()),
                            np.min(exp_2.loss_d.numpy()),
                            exp_2.loss.numpy(),
                            np.min(exp_2.loss_r.numpy()) / (np.max(x_r) - np.min(x_r)),
                            np.min(exp_2.loss_d.numpy()) / (np.max(x_d) - np.min(x_d)),
                            exp_2.loss.numpy() / (np.max(x) - np.min(x))
                        ]
                    },
                    index=['RMSE_r', 'RMSE_d', 'RMSE', 'RRMSE_r', 'RRMSE_d', 'RRMSE']
                )
                print(df_e)
            elif loss == 'rrmse_range' or loss == 'rrmse_mean' or loss == 'rrmse_euclidean' or loss == 'rrmse_percent':
                df_e = pd.DataFrame(
                    {
                        'value': [
                            np.min(exp_2.loss_r.numpy()),
                            np.min(exp_2.loss_d.numpy()),
                            exp_2.loss.numpy(),
                        ]
                    },
                    index=['RRMSE_r', 'RRMSE_d', 'RRMSE']
                )
                print(df_e)

    # Plot
    fig = plt.figure(figsize=(12, 5.4))

    if os.path.isfile(file_iv):
        x_init = (exp_1.current[0] / exp_1.voltage[0] - exp_1.G_on) / (exp_1.G_off - exp_1.G_on)
        x_init = x_init if x_init > 0 else 0
        x_init = x_init if x_init < 1 else 1
        mem_x, mem_c = exp_1.Memristor_conductance_model(alpha_off, alpha_on, x_init, exp_1.voltage)
        current_fit = np.array(mem_c) * np.array(exp_1.voltage)
        ax1 = fig.add_subplot(121)
        ax1.plot(exp_1.voltage, current_fit, c='b')
        ax1.scatter(exp_1.voltage, exp_1.current, c='r')
        ax1.set_xlabel('Voltage (V)')
        ax1.set_ylabel('Current (A)')
        ax1.set_title('I-V Curve')

    if os.path.isfile(file_c):
        plot_x = np.arange(exp_2.points_r + exp_2.points_d)
        ax2 = fig.add_subplot(122)
        ax2.plot(plot_x, memx_total, c='b')
        for i in range(exp_2.device_nums):
            ax2.scatter(plot_x, x_total.T[i], c='r', s=0.1, alpha=0.3)
        ax2.set_xlabel('points')
        ax2.set_ylabel('x')
        ax2.set_title('Conductance Curve')

    plt.tight_layout()
    plt.savefig("Baseline Model.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
