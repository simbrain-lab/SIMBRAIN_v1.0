import os
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
sys.path.append('../../../')
from simbrain.Fitting_Functions.conductance_fitting import Conductance
from simbrain.Fitting_Functions.variation_fitting import Variation



def main():
    # Fit
    with open("../../../memristor_data/my_memristor.json") as f:
        dict = json.load(f)
    dict.update(
        {
            'v_off': 1.5,
            'v_on': -1.5,
            'G_off': None,
            'G_on': None,
            'alpha_off': None,
            'alpha_on': None,
            'k_off': None,
            'k_on': None,
            'P_off': None,
            'P_on': None,
            'delta_t': 20 * 1e-3,
            'duty_ratio': 0.5
        }
    )
    file = "../../../reference_memristor_data/conductance_deletehead.xlsx"
    file_G = "../../../reference_memristor_data/G_variation.xlsx"

    if os.path.isfile(file_G):  
        data = pd.DataFrame(pd.read_excel(
            file_G,
            sheet_name='Sheet1',
            header=None,
            index_col=None
        ))
        data.columns = ['G_off', 'G_on']
        device_nums = data.shape[0]

        G_off_list = np.array(data['G_off'])
        G_on_list = np.array(data['G_on'])
        G_off = np.mean(G_off_list)
        G_on = np.mean(G_on_list)
    else:
        data = pd.DataFrame(pd.read_excel(
            file,
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
    dict.update(
        {
            'G_off': G_off,
            'G_on': G_on
        }
    )

    P_off_list = np.zeros(device_nums)
    P_on_list = np.zeros(device_nums)

    _, Goff_sigma, _, Gon_sigma = Variation(
        file,
        G_off_list,
        G_on_list,
        P_off_list,
        P_on_list,
        dict
    ).d2d_G_fitting()
    dict.update(
        {
            "Goff_sigma": Goff_sigma,
            "Gon_sigma": Gon_sigma,
        }
    )

    alpha_off, alpha_on = 5, 5
    dict.update(
        {
            "alpha_off": alpha_off,
            "alpha_on": alpha_on
        }
    )
    loss = 'rmse'
    exp_0 = Conductance(file, dict)
    P_off, P_on, k_off, k_on, _, _ = exp_0.fitting(loss_option=loss)
    dict.update(
        {
            "P_off": P_off,
            "P_on": P_on,
            'k_off': k_off,
            'k_on': k_on
        }
    )

    P_off_list, P_on_list = exp_0.mult_P_fitting(G_off_list, G_on_list, loss_option=loss)

    exp = Variation(
        file,
        G_off_list,
        G_on_list,
        P_off_list,
        P_on_list,
        dict
    )
    _, Poff_sigma, _, Pon_sigma = exp.d2d_P_fitting()
    dict.update(
        {
            "Poff_sigma": Pon_sigma,
            "Pon_sigma": Poff_sigma
        }
    )
    sigma_relative, sigma_absolute = exp.c2c_fitting(cluster_option='ew')
    dict.update(
        {
            "P_off": P_off,
            "P_on": P_on,
            "sigma_relative": sigma_relative,
            "sigma_absolute": sigma_absolute
        }
    )
    # Output
    df = pd.DataFrame(
        {'value': [Goff_sigma, Gon_sigma, Poff_sigma, Pon_sigma, sigma_relative, sigma_absolute]},
        index=['Goff_sigma', 'Gon_sigma', 'Poff_sigma', 'Pon_sigma', 'sigma_relative', 'sigma_absolute']
    )
    print(df)
    print('R_square:', exp.R_square)
    conductance = np.array(exp_0.data[:])[:, 2:] / exp_0.read_voltage
    x = (conductance - G_on) / (G_off - G_on)
    if loss == 'rmse':
        rrmse = exp_0.loss / (np.max(x) - np.min(x))
    else:
        rrmse = exp_0.loss
    print('RRMSE:', rrmse.item())
    with open("fitting_record.json", "w") as f:
        json.dump(dict, f, indent=2)
    # Output d2d variation data
    data_G = pd.DataFrame(
        {
            'G_off': G_off_list,
            'G_on': G_on_list,
        }
    )
    # data_G.to_excel("../Fitting_results/G_variation.xlsx", index=False, header=False)
    data_P = pd.DataFrame(
        {
            'P_off': P_off_list,
            'P_on': P_on_list,
        }
    )
    # data_P.to_excel("../Fitting_results/P_variation.xlsx", index=False, header=False)

    # Plot
    plot_x_1 = np.linspace(exp.G_off * (1 - 3 * Goff_sigma), exp.G_off * (1 + 3 * Goff_sigma))
    plot_x_2 = np.linspace(exp.G_on * (1 - 3 * Gon_sigma), exp.G_on * (1 + 3 * Gon_sigma))
    plot_y_1 = norm.pdf(plot_x_1, exp.G_off, exp.G_off * Goff_sigma)
    plot_y_2 = norm.pdf(plot_x_2, exp.G_on, exp.G_on * Gon_sigma)

    fig = plt.figure(figsize=(12, 16))
    ax1 = fig.add_subplot(321)
    ax1.hist(exp.G_off_variation, bins=10, density=True)
    ax1.plot(plot_x_1, plot_y_1, c='r')
    ax1.set_xlabel('G_off')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('D2D Variation (G_off)')
    ax2 = fig.add_subplot(322)
    ax2.hist(exp.G_on_variation, bins=10, density=True)
    ax2.plot(plot_x_2, plot_y_2, c='orange')
    ax2.set_xlabel('G_on')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('D2D Variation (G_on)')

    plot_x_1 = np.linspace(exp.P_off * (1 - 3 * Poff_sigma), exp.P_off * (1 + 3 * Poff_sigma))
    plot_x_2 = np.linspace(exp.P_on * (1 - 3 * Pon_sigma), exp.P_on * (1 + 3 * Pon_sigma))
    plot_y_1 = norm.pdf(plot_x_1, exp.P_off, exp.P_off * Poff_sigma)
    plot_y_2 = norm.pdf(plot_x_2, exp.P_on, exp.P_on * Pon_sigma)

    ax3 = fig.add_subplot(323)
    ax3.hist(exp.P_off_variation, bins=10, density=True)
    ax3.plot(plot_x_1, plot_y_1, c='green')
    ax3.set_xlabel('P_off')
    ax3.set_ylabel('Probability Density')
    ax3.set_title('D2D Variation (P_off)')
    ax4 = fig.add_subplot(324)
    ax4.hist(exp.P_on_variation, bins=10, density=True, color='orange')
    ax4.plot(plot_x_2, plot_y_2, c='red')
    ax4.set_xlabel('P_on')
    ax4.set_ylabel('Probability Density')
    ax4.set_title('D2D Variation (P_on)')

    z = np.array([sigma_relative ** 2 * 2 / np.pi, sigma_absolute ** 2 * 2 / np.pi])
    p = np.poly1d(z)
    plot_x = exp.x_mean
    plot_y = p(plot_x)

    ax5 = fig.add_subplot(325)
    ax5.scatter(np.square(exp.memx_total), np.square(exp.variation_x), c='r')
    ax5.set_xlabel('x')
    ax5.set_ylabel('Variation')
    ax5.set_title('C2C Variation')
    ax6 = fig.add_subplot(326)
    ax6.scatter(np.square(plot_x), np.square(exp.var_x_average), c='r')
    ax6.plot(plot_x, plot_y, c='b')
    ax6.set_xlabel('x')
    ax6.set_ylabel('Variation_mean')
    ax6.set_title('C2C Variation fitting')

    plt.tight_layout()
    plt.savefig("Variation.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
