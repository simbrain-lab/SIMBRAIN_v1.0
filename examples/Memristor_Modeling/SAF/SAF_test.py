import json
import sys
sys.path.append('../../../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simbrain.Fitting_Functions.stuck_at_fault_fitting import StuckAtFault


def main():
    # Fit
    # Pre-deployment SAF
    with open("../../../memristor_data/my_memristor.json") as f:
        dict = json.load(f)
    file = "../../../reference_memristor_data/saf_data.xlsx"
    exp = StuckAtFault(file)
    SAF_lambda, SAF_ratio = exp.pre_deployment_fitting()
    dict.update(
        {
            "SAF_lambda": SAF_lambda,
            "SAF_ratio": SAF_ratio
        }
    )
    # Post-deployment SAF
    SAF_delta = exp.post_deployment_fitting()
    dict.update(
        {
            "SAF_delta": SAF_delta
        }
    )

    # Output
    df = pd.DataFrame(
        {'value': [SAF_lambda, SAF_ratio, SAF_delta]},
        index=['SAF_lambda', 'SAF_ratio', 'SAF_delta']
    )
    print(df)
    with open("fitting_record.json", "w") as f:
        json.dump(dict, f, indent=2)

    # Plot
    blue = (47 / 255, 130 / 255, 189 / 255)
    orange = (236 / 255, 111 / 255, 36 / 255)
    brown = (135 / 255, 78 / 255, 4 / 255)
    green = (98 / 255, 149 / 255, 61 / 255)
    yellow = (255 / 255, 183 / 255, 11 / 255)
    colors = [blue, yellow, orange]

    fig = plt.figure(figsize=(12, 5.4))

    # pre-deployment饼图
    SA0 = SAF_lambda * SAF_ratio / (1 + SAF_ratio)
    SA1 = SAF_lambda / (1 + SAF_ratio)
    Work = 1 - SAF_lambda

    ax1 = fig.add_subplot(121)
    ax1.pie(
        np.array([SA0, SA1, Work]),
        explode=(0.1, 0.1, 0),
        labels=['SA0', 'SA1', 'Work'],
        colors=colors,
        autopct='%1.1f%%'
    )
    ax1.set_title('Pre-deployment SAF')

    # post-deployment时间趋势
    mem_t = exp.data.columns[1] - exp.data.columns[0]
    SAF = []
    for i in range(exp.data.shape[1]):
        SAF.append(np.count_nonzero(exp.data[mem_t * i]) / exp.data.shape[0])

    z = np.array([SAF_delta, SAF_lambda])
    p = np.poly1d(z)
    plot_y = p(exp.data.columns)

    ax2 = fig.add_subplot(122)
    ax2.scatter(exp.data.columns, SAF, c=orange, s=250)
    ax2.plot(exp.data.columns, plot_y, c=blue, linewidth=2.5)
    ax2.set_xlabel('Time(s)')
    ax2.set_ylabel('Stuck At Fault')
    ax2.set_title('Post-deployment SAF')

    plt.tight_layout()
    plt.savefig("stuck at fault.png")
    plt.show()


if __name__ == "__main__":
    main()
