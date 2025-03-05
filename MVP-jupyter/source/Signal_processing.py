import pandas as pd
from scipy import signal
import numpy as np

import re

from Peaks_processing import Signal_meta, proc_slices, get_slices, Slice


def filt_signal(arr: np.ndarray, N: int, W: float) -> np.ndarray:
    '''

    :param arr:
    :param N:
    :param W:
    :return:
    '''
    b, a = signal.butter(N, W)
    return signal.filtfilt(b, a, arr)


def smooth_rect(y: np.ndarray, box_pts: int, ro: float = 1.) -> np.ndarray:
    '''

    :param y:
    :param box_pts:
    :param ro:
    :return:
    '''
    box = np.ones(box_pts) / (box_pts * ro)
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def smooth_gauss(y: np.ndarray, box_pts: int, sigma: float = 1.) -> np.ndarray:
    '''

    :param y:
    :param box_pts:
    :param sigma:
    :return:
    '''
    x = np.arange(box_pts) - box_pts / 2
    box = 1 / sigma / (2 * np.pi) ** 0.5 * np.exp(-x ** 2 / 2 / sigma ** 2)
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def smooth_steklov(y: np.ndarray, box_pts: int, ro: float = 0.1) -> np.ndarray:
    '''

    :param y:
    :param box_pts:
    :param ro:
    :return:
    '''
    c = 1 / 0.4439938161681
    x_ro = (np.arange(box_pts) - box_pts / 2) / (box_pts / 2 * ro)
    box = np.zeros(box_pts)
    box[abs(x_ro) < 1] = 1 / (box_pts / 2 * ro) * c * np.exp(1 / (x_ro[abs(x_ro) < 1] ** 2 - 1))
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def get_ind_fromColumns(columns: list) -> list:
    return [int(re.search(r"ch\d", name).group(1)) for name in columns]


def dbs_A_dFi(df: pd.DataFrame,
              w: float, smooth_length: int,
              ro_A: int = 0.85, ro_dfi: int = 0.5,
              smoothing: str = "steklov") -> None:
    '''

    :param df:
    :param w:
    :param smooth_length:
    :param ro_dfi:
    :param ro_A:
    :param smoothing:
    :return:
    '''
    channels = get_ind_fromColumns(df.columns)

    if smoothing == "rect":
        smoothing_f = smooth_rect
    elif smoothing == "gauss":
        smoothing_f = smooth_steklov
    elif smoothing == "steklov":
        smoothing_f = smooth_steklov
    else:
        raise ValueError("smoothing must be either 'rect', 'gauss' or 'steklov'")

    for i in range(0, len(channels), 2):
        ind = channels[i]
        i_data = df[f"ch{ind}"].to_numpy()
        i_data /= np.linalg.norm(i_data)
        q_data = df[f"ch{ind + 1}"].to_numpy()
        q_data /= np.linalg.norm(q_data)

        c_data = filt_signal(i_data, 5, w) + filt_signal(q_data, 5, w) * 1j

        fi_data = np.unwrap(np.angle(c_data))

        df[f"ch{ind}_A"] = smoothing_f(np.abs(c_data), smooth_length, ro_A)
        df[f"ch{ind}_dfi"] = smoothing_f(np.concatenate([np.diff(fi_data), [0]]), smooth_length, ro_dfi)


def get_marked_signal(signal: pd.Series, meta_signal: Signal_meta,
                      w: float,
                      stats_params: list, f_proc: bool = False) -> np.ndarray:
    '''

    :param signal:
    :param meta_signal:
    :param w:
    :param stats_params:
    :param f_proc:
    :return:
    '''
    if signal is None:
        raise ValueError('Please provide sxr')
    if meta_signal is None:
        raise ValueError('Please provide meta_signal')

    signal_d1 = np.diff(signal)
    signal_d1_f = filt_signal(signal_d1, 5, w)

    meta_signal.set_statistics(signal, signal_d1_f,
                               stats_params[0], stats_params[1],
                               d_std_bottom_edge=stats_params[2], d_std_top_edge=stats_params[3])

    mark_data = np.zeros(signal_d1_f.shape)
    mark_data[abs(signal_d1_f - meta_signal.d_q) > meta_signal.d_std * meta_signal.d_std_bottom] = 1

    return proc_slices(mark_data, signal, signal_d1_f, meta_signal) if f_proc else mark_data


def get_shoot_slices(d_alpha: pd.Series = None, sxr: pd.Series = None, verbose: int = 0) -> np.ndarray:
    '''

    :param d_alpha:
    :param sxr:
    :param verbose:
    :return:
    '''
    if sxr is None:
        raise ValueError('Please provide sxr')

    meta_sxr = Signal_meta(chanel_name="sxr", processing_flag=True)
    meta_sxr.set_edges(length_edge=5, distance_edge=30, scale=0, step_out=20)

    mark_sxr = get_marked_signal(sxr, meta_sxr, 0.05, [0.8, 0.8, 7., 13.], f_proc=True)

    sxr_slices = get_slices(mark_sxr)
    slices_edges = []
    step_out = 50

    len_top = 5000
    len_width = 3000
    len_step = 2000

    # get slices btw sxr falls
    slices_edges.append([sxr_slices[0].r + step_out, 0])
    for i in range(1, len(sxr_slices)):
        cur_l_edge = slices_edges[-1][0]
        while sxr_slices[i].l - step_out - cur_l_edge > len_top:
            slices_edges[-1][1] = cur_l_edge + len_width
            slices_edges.append([cur_l_edge + len_width, 0])
            cur_l_edge += len_step

        slices_edges[-1][1] = sxr_slices[i].l - step_out
        slices_edges.append([sxr_slices[i].r + step_out, 0])

    slices_edges[-1][1] = min(sxr_slices[-1].r + 1000, sxr.shape[0] - step_out)
    slices_edges = np.array(slices_edges)
    if verbose:
        print("Slices from splitting SXR falls:", len(slices_edges))

    first_slice = Slice(150, sxr_slices[0].l + step_out)

    if verbose:
        print("First slice:", first_slice)

    meta_da = Signal_meta(chanel_name="da", processing_flag=True)
    meta_da.set_edges(length_edge=50, distance_edge=100)

    mark_data = get_marked_signal(d_alpha, meta_da, 0.1, [0.7, 0.7, 2., 1.], f_proc=True)[first_slice.l: first_slice.r]

    da_slices = get_slices(mark_data)

    first_slice_edges = []
    dist_top = 1000
    step_out = 100

    first_slice_edges.append([max(0, da_slices[0].l - step_out), da_slices[0].r + step_out])
    for i in range(len(da_slices)):
        if da_slices[i].l - first_slice_edges[-1][1] > dist_top:
            first_slice_edges.append([da_slices[i].l - step_out, da_slices[i].l])
        elif first_slice_edges[-1][1] - first_slice_edges[-1][0] > len_top:
            first_slice_edges.append([first_slice_edges[-1][1] - step_out, first_slice_edges[-1][1]])
        first_slice_edges[-1][1] = da_slices[i].r + step_out

    first_slice_edges = np.array(first_slice_edges) + 150

    if verbose:
        print("Slices from splitting first slice:", len(first_slice_edges))

    return np.concatenate([first_slice_edges, slices_edges]).astype(np.int64)
