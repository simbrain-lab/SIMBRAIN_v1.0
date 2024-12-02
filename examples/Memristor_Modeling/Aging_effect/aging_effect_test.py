import json
import shutil
import sys
sys.path.append('../../../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simbrain.Fitting_Functions.aging_effect_fitting import AgingEffect


def main():
    # Fit
    with open("../../../memristor_data/my_memristor.json") as f:
        dict = json.load(f)
    file = "../../../reference_memristor_data/aging_effect.xlsx"
    exp = AgingEffect(file, dict)
    Aging_off_1, Aging_on_1 = exp.fitting_equation1()
    Aging_off_2, Aging_on_2 = exp.fitting_equation2()
    dict.update(
        {
            'Aging_off': Aging_off_2,
            'Aging_on':  Aging_on_2
        }
    )

    # Output
    print('Equation 1:')
    df = pd.DataFrame(
        {'value': [Aging_off_1,  Aging_on_1]},
        index=['Aging_off_1', 'Aging_on_1']
    )
    print(df)
    print('Equation 2:')
    df = pd.DataFrame(
        {'value': [Aging_off_2,  Aging_on_2]},
        index=['Aging_off_2', 'Aging_on_2']
    )
    print(df)
    with open("fitting_record.json", "w") as f:
        json.dump(dict, f, indent=2)

    # Plot
    # # Four subgraphs
    # fig = plt.figure(figsize=(12.8, 7.2))
    # plot_x = np.logspace(-3, 4, 1000, base=10)
    #
    # ax1 = fig.add_subplot(221)
    # ax1.scatter(exp.CYCLE, exp.HCS, c='red', s=0.1)
    # exp.G_0 = exp.G_0_L
    # ax1.semilogx(plot_x, exp.equation_1(plot_x, Aging_off_1))
    # ax1.set_xlim(1e-3, 1e4)
    # ax1.set_xlabel('Time(s) * 10^3')
    # ax1.set_ylabel('Conductance(S)')
    # ax1.set_title('Aging effect G_off(equation 1)')
    #
    # ax2 = fig.add_subplot(222)
    # ax2.scatter(exp.CYCLE, exp.LCS, c='red', s=0.1)
    # exp.G_0 = exp.G_0_H
    # ax2.semilogx(plot_x, exp.equation_1(plot_x,  Aging_on_1))
    # ax2.set_xlim(1e-3, 1e4)
    # ax2.set_xlabel('Time(s) * 10^3')
    # ax2.set_ylabel('Conductance(S)')
    # ax2.set_title('Aging effect G_on(equation 1)')
    #
    # ax3 = fig.add_subplot(223)
    # ax3.scatter(exp.CYCLE, exp.HCS, c='red', s=0.1)
    # ax3.semilogx(plot_x, exp.equation_2(plot_x, Aging_off_2, b_off))
    # ax3.set_xlim(1e-3, 1e4)
    # ax3.set_xlabel('Time(s) * 10^3')
    # ax3.set_title('Aging effect G_off(equation 2)')
    #
    # ax4 = fig.add_subplot(224)
    # ax4.scatter(exp.CYCLE, exp.LCS, c='red', s=0.1)
    # ax4.semilogx(plot_x, exp.equation_2(plot_x,  Aging_on_2, b_on))
    # ax4.set_xlim(1e-3, 1e4)
    # ax4.set_xlabel('Time(s) * 10^3')
    # ax4.set_title('Aging effect G_on(equation 2)')

    # Two subgraphs
    fig = plt.figure(figsize=(15, 6))
    plot_x = np.logspace(-3, -0.3, 1000, base=10)
    ax1 = fig.add_subplot(121)
    exp.G_0 = exp.G_off_init
    ax1.semilogx(plot_x, np.exp(exp.equation_1_log(plot_x, Aging_off_1)))
    ax1.semilogx(plot_x, exp.equation_2(plot_x, Aging_off_2))
    ax1.scatter(exp.mem_t, exp.G_off, c='red', s=0.1)
    ax1.legend(['Equation 1', 'Equation 2'])
    ax1.set_xlabel('Time(s)')
    ax1.set_ylabel('Conductance(S)')
    ax1.set_title('Aging effect (G_off)')

    ax2 = fig.add_subplot(122)
    exp.G_0 = exp.G_on_init
    ax2.semilogx(plot_x, np.exp(exp.equation_1_log(plot_x,  Aging_on_1)))
    ax2.semilogx(plot_x, exp.equation_2(plot_x,  Aging_on_2))
    ax2.scatter(exp.mem_t, exp.G_on, c='red', s=0.1)
    ax2.legend(['Equation 1', 'Equation 2'])
    ax2.set_xlabel('Time(s)')
    ax2.set_ylabel('Conductance(S)')
    ax2.set_title('Aging effect (G_on)')

    plt.tight_layout()
    plt.savefig("Aging Effect.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
