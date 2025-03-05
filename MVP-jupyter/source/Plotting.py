import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

from source.Signal_processing import smooth_steklov, filt_signal, get_ind_fromColumns
from source.Peaks_processing import get_d1_crosses, get_groups_from_signal, get_time_delta

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

colors = ['gold', 'brown', 'black', 'seagreen', 'cyan', 'yellow', 'gray', 'violet', 'royalblue', 'sandybrown', 'grey',
          'indigo', 'rosybrown', 'darkviolet', 'coral', 'pink', 'magenta', 'red', 'springgreen', 'darkblue', 'silver',
          'seashell', 'green', 'navy', 'purple', 'sienna', 'chocolate', 'orange', 'blue']


def plot_shoot(F_ID: int, res_slices_edges: np.ndarray,
               start_ind: int, bottom_width: int,
               df: pd.DataFrame, dbs_df: pd.DataFrame,
               smooth_length: int,
               df_groups_arr: list, df_points_arr: list) -> None:
    '''

    :param F_ID:
    :param res_slices_edges:
    :param start_ind:
    :param bottom_width:
    :param df:
    :param dbs_df:
    :param smooth_length:
    :param df_groups_arr:
    :param df_points_arr:
    :return:
    '''

    channels = get_ind_fromColumns(list(dbs_df.columns))
    ind = start_ind
    while True:

        while ind >= len(res_slices_edges):
            print(f"The end! Press 'Enter' to stop or input slice number (from 1 to {len(res_slices_edges)}): ")
            ind = input()
            if ind == '':
                break
            else:
                ind = int(ind) - 1

        l_edge, r_edge = res_slices_edges[ind]
        plot_l_edge, plot_r_edge = l_edge, r_edge

        if plot_r_edge - plot_l_edge < bottom_width:
            increasing_d = bottom_width - (plot_r_edge - plot_l_edge)
            plot_l_edge -= increasing_d
            plot_r_edge += increasing_d

        fig, axs = plt.subplots(nrows=8, gridspec_kw={'hspace': 0.2})

        fig.set_figwidth(20)
        fig.set_figheight(20)

        axs[0].set_title(f"#{F_ID}")

        d_alpha = df.d_alpha.to_numpy()
        d_alpha_f = filt_signal(np.diff(d_alpha), 5, 0.1)
        d_alpha_d2f = filt_signal(np.diff(d_alpha_f), 5, 0.1)

        axs[0].plot(range(plot_l_edge, plot_r_edge), d_alpha[plot_l_edge:plot_r_edge],
                    label="D-alpha", alpha=0.8, zorder=2)

        edges_y = (d_alpha[plot_l_edge] + d_alpha[plot_r_edge]) / 2
        axs[0].scatter([l_edge, r_edge], [edges_y, edges_y], s=1000, color="black", marker="|",
                       zorder=1)

        start_time = time.time()
        coef = 1.
        x = get_d1_crosses(d_alpha_f, d_alpha_d2f, l_edge, r_edge, d1_coef=coef)
        print(f"\n------\n------\n\n{ind + 1}/{res_slices_edges.shape[0]} - Slice ({l_edge / 1e3}, {r_edge / 1e3}) ms" +
              f" - {len(x)} peaks - {(time.time() - start_time) * 1e3:.3f} ms")

        res_groups_peaks = []

        if len(x) > 1:
            print(f"Start prossecing peaks ...", end=" ")  #
            start_time = time.time()
            res_groups_peaks = get_groups_from_signal(d_alpha, d_alpha_f, d_alpha_d2f, l_edge, r_edge)
            # print("- logg: ", res_groups_peaks)
            print(f"- Tooks: {(time.time() - start_time) * 1e3:.3f} ms")
            for g_i in range(len(res_groups_peaks)):
                points = res_groups_peaks[g_i]
                c = colors[g_i % len(colors)]
                axs[0].scatter(points, d_alpha[points] + (2 * (g_i % 2) - 1) * 0.05, s=10, color=c, zorder=0)

                m_d = get_time_delta(points) / 1e3
                std_d = 0

                for p_i in range(1, len(points)):
                    std_d += (m_d - (points[p_i] - points[p_i - 1]) / 1e3) ** 2
                std_d = (std_d / len(points) / (len(points) - 1)) ** .5
                print(f"{g_i + 1}/{len(res_groups_peaks)} Group of peaks [{np.argwhere(x == points[0])[0, 0]}-" +
                      f"{np.argwhere(x == points[-1])[0, 0]}] ({c}) - {len(points)} peaks in group - " +
                      f"mean delta: {m_d:.3f} ms - freq: {1 / m_d:.3f} +- {std_d / (m_d ** 2):.3f} kHz")

            for p_i in range(len(x)):
                num = p_i
                d = 0
                while num > 10:
                    num = num // 10
                    d += 1
                axs[0].annotate(p_i, (x[p_i] - (25) * (r_edge - l_edge) / 5000 - 15 * d, d_alpha[x[p_i]] + 0.03))
                for ax in axs:
                    ax.axvline(x[p_i], linestyle=':', color='k', alpha=0.7)
        elif len(x) == 1:
            axs[0].annotate(0, (x[0] - (25) * (r_edge - l_edge) / 5000, d_alpha[x[0]] + 0.03))
            axs[0].scatter(x, d_alpha[x], s=20, color="black")
            for ax in axs:
                ax.axvline(x[0], linestyle=':', color='k', alpha=0.7)

        for i in channels[::2]:  # channels[::2]
            time_mask = np.array((dbs_df.t * 1e6 >= plot_l_edge) & (dbs_df.t * 1e6 <= plot_r_edge))
            axs[1].plot(np.linspace(plot_l_edge, plot_r_edge, np.count_nonzero(time_mask)),
                        normalize(dbs_df[f"ch{i}"].to_numpy()[:, np.newaxis], axis=0).ravel()[time_mask],
                        label=f"ch{i}", alpha=0.8)
            axs[2].plot(np.linspace(plot_l_edge, plot_r_edge, np.count_nonzero(time_mask)),
                        normalize(dbs_df[f"ch{i + 1}"].to_numpy()[:, np.newaxis], axis=0).ravel()[time_mask],
                        label=f"ch{i}", alpha=0.8)
            axs[3].plot(np.linspace(plot_l_edge, plot_r_edge, np.count_nonzero(time_mask)),
                        dbs_df[f"ch{i}_A"][time_mask], label=f"ch{i}")
            axs[4].plot(np.linspace(plot_l_edge, plot_r_edge, np.count_nonzero(time_mask)),
                        dbs_df[f"ch{i}_dfi"][time_mask], label=f"ch{i}", alpha=0.8)

        axs[-3].plot(range(plot_l_edge, plot_r_edge), df.mhd[plot_l_edge:plot_r_edge], label="Abs MHD")
        axs[-3].plot(range(plot_l_edge, plot_r_edge), smooth_steklov(df.mhd, smooth_length)[plot_l_edge:plot_r_edge],
                     label="Smooth abs MHD")

        axs[-2].plot(range(plot_l_edge, plot_r_edge), df.nl[plot_l_edge:plot_r_edge], label="NL")
        axs[-2].plot(range(plot_l_edge, plot_r_edge), smooth_steklov(df.nl.to_numpy(), smooth_length)[plot_l_edge:plot_r_edge],
                     label="Smooth NL")

        axs[-1].plot(range(plot_l_edge, plot_r_edge), df.sxr[plot_l_edge:plot_r_edge], label="SXR")
        axs[-1].plot(range(plot_l_edge, plot_r_edge), smooth_steklov(df.sxr, smooth_length)[plot_l_edge:plot_r_edge],
                     label="Smooth SXR")

        plot_ylabels = [r"$D_\alpha$", "DBS I", "DBS Q", "DBS A", r"DBS $\partial\phi$", "Abs Rad&Vert MHD", "Nl",
                        "SXR"]  # , "MHD 4"
        for ax_i, ax in enumerate(axs):
            ax.set_ylabel(plot_ylabels[ax_i])
            ax.grid(which='major', color='#DDDDDD', linewidth=0.9)
            ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(AutoMinorLocator(10))
            ax.legend(loc='upper right')

        plt.show()

        mode = input("Input mode [manual (input points) - 0 | auto (input group) - 1 | " +
                     "input index - -1 | continue - 'Enter']: ")
        while mode != "" and int(mode) >= 0:
            mode = int(mode)
            if mode == 0:
                points_ind = list(map(int, input("Input point indexes:\n").strip().split()))
                points = x[points_ind]
            else:
                gr_ind = int(input(f"Input group number (from 1 to {len(res_groups_peaks)}): ").strip().split()[0]) - 1
                points = res_groups_peaks[gr_ind]

            # mean delta in group
            m_d = get_time_delta(points) / 1e3

            # delta std
            std_d = 0
            for p_i in range(len(points)):
                if p_i > 0:
                    std_d += (m_d - (points[p_i] - points[p_i - 1]) / 1e3) ** 2
            std_d = (std_d / len(points) / (len(points) - 1)) ** .5

            print(f"Group stats: n: {len(points)}, fr: ({1 / m_d:.3f} +- {std_d / (m_d ** 2):.3f}) kHZ")

            mark = input("Input mark of the group (string: eho|lco|delm)")

            df_groups_arr.append([points[0] - 50, points[-1] + 50, len(points), 1 / m_d, std_d / (m_d ** 2), mark])

            for p in points:
                df_points_arr.append([df.t[p], 1 / m_d, mark])

            print("------")
            mode = input("Input mode [manual (input points) - 0 | auto (input group) - 1 | " +
                         "input index - -1 | continue - 'Enter']: ")

        if mode != "" and int(mode) < 0:
            ind = int(input(f"Input index (from 1 to {len(res_slices_edges)}): ")) - 1
        else:
            ind += 1
