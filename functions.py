import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from read_waveform import read_waveform as rw


def calculate_average(t, v):
    v_sum = 0
    idx = np.abs(t - 0).argmin()
    if idx > len(t) / 2:
        idx1 = int(.1 * len(t))
        idx2 = int(.4 * len(t))
    else:
        idx1 = int(.6 * len(t))
        idx2 = int(.9 * len(t))
    for i in range(idx1, idx2):
        v_sum += v[i]
    average = v_sum / (idx2 - idx1)
    return average


def calculate_charge(t, v, r):
    avg = calculate_average(t, v)
    vsum = 0
    idx1 = np.inf
    idx2 = np.inf

    idx_min_val = np.where(v == min(v))
    time_min_val = t[idx_min_val]
    min_time = time_min_val[0]

    tvals = np.linspace(t[0], t[len(t) - 1], int(2e6))
    tvals1 = np.linspace(t[0], min_time, int(2e6))
    tvals2 = np.linspace(min_time, t[len(t) - 1], int(2e6))
    vvals = np.interp(tvals, t, v)
    vvals1 = np.interp(tvals1, t, v)
    vvals2 = np.interp(tvals2, t, v)
    vvals1_flip = np.flip(vvals1)
    difference_value1 = vvals1_flip - avg
    difference_value2 = vvals2 - avg

    for i in range(0, len(difference_value1) - 1):
        if difference_value1[i] >= 0:
            idx1 = len(difference_value1) - i
            break
    if idx1 == np.inf:
        idx1 = len(difference_value1) - np.argmin(np.abs(difference_value1))
    for i in range(0, len(difference_value2) - 1):
        if difference_value2[i] >= 0:
            idx2 = i
            break
    if idx2 == np.inf:
        idx2 = np.argmin(np.abs(difference_value2))

    diff_val1 = np.abs(tvals - tvals1[idx1])
    diff_val2 = np.abs(tvals - tvals2[idx2])
    index1 = np.argmin(diff_val1)
    index2 = np.argmin(diff_val2)
    t1 = tvals[index1]
    t2 = tvals[index2]
    for i in range(index1.item(), index2.item()):
        vsum += vvals[i]
    charge = -1 * (t2 - t1) * vsum / ((index2 - index1) * r)
    return t1, t2, charge


def calculate_amp(t, v):
    avg = calculate_average(t, v)
    min_val = np.amin(v)
    amp = avg - min_val
    return amp


def calculate_fwhm(t, v, half_max):
    tvals = np.linspace(t[0], t[len(t) - 1], int(2e6))
    vvals = np.interp(tvals, t, v)
    difference_value = np.abs(vvals - half_max)
    differential = np.diff(vvals)
    for i in range(0, len(differential) - 1):
        if differential[i] < 0:
            difference_value[i] = np.inf
    idx = np.argmin(difference_value)
    half_max_time = tvals[idx]
    return half_max_time


def rise_time(t, v, r):
    avg = calculate_average(t, v)
    t1, t2, charge = calculate_charge(t, v, r)
    idx_min_val = np.where(v == min(v))
    time_min_val = t[idx_min_val]
    min_time = time_min_val[0]

    val10 = .1 * (min(v) - avg)
    val20 = 2 * val10
    val80 = 8 * val10
    val90 = 9 * val10
    tvals = np.linspace(t1, min_time, int(2e6))
    vvals = np.interp(tvals, t, v)
    difference_value10 = np.abs(vvals - val10)
    difference_value20 = np.abs(vvals - val20)
    difference_value80 = np.abs(vvals - val80)
    difference_value90 = np.abs(vvals - val90)
    index10 = np.argmin(difference_value10)
    index20 = np.argmin(difference_value20)
    index80 = np.argmin(difference_value80)
    index90 = np.argmin(difference_value90)
    time10 = tvals[index10]
    time20 = tvals[index20]
    time80 = tvals[index80]
    time90 = tvals[index90]
    rise_time1090 = time90 - time10
    rise_time2080 = time80 - time20
    rise_time1090 = float(format(rise_time1090, '.2e'))
    rise_time2080 = float(format(rise_time2080, '.2e'))
    return rise_time1090, rise_time2080


def fall_time(t, v, r):
    avg = calculate_average(t, v)
    t1, t2, charge = calculate_charge(t, v, r)
    idx_min_val = np.where(v == min(v))
    time_min_val = t[idx_min_val]
    min_time = time_min_val[0]

    val10 = .1 * (min(v) - avg)
    val20 = 2 * val10
    val80 = 8 * val10
    val90 = 9 * val10
    tvals = np.linspace(min_time, t2, int(2e6))
    vvals = np.interp(tvals, t, v)
    difference_value10 = np.abs(vvals - val10)
    difference_value20 = np.abs(vvals - val20)
    difference_value80 = np.abs(vvals - val80)
    difference_value90 = np.abs(vvals - val90)
    index10 = np.argmin(difference_value10)
    index20 = np.argmin(difference_value20)
    index80 = np.argmin(difference_value80)
    index90 = np.argmin(difference_value90)
    time10 = tvals[index10]
    time20 = tvals[index20]
    time80 = tvals[index80]
    time90 = tvals[index90]
    fall_time1090 = time10 - time90
    fall_time2080 = time20 - time80
    fall_time1090 = float(format(fall_time1090, '.2e'))
    fall_time2080 = float(format(fall_time2080, '.2e'))
    return fall_time1090, fall_time2080


def make_arrays(save_shift, start, end, nhdr, r, half_max):
    t1_array = np.array([])
    t2_array = np.array([])
    charge_array = np.array([])
    amplitude_array = np.array([])
    fwhm_array = np.array([])
    rise1090_array = np.array([])
    rise2080_array = np.array([])
    fall1090_array = np.array([])
    fall2080_array = np.array([])

    for i in range(start, end + 1):
        file_name = str(save_shift / 'D1--waveforms--%05d.txt') % i
        if os.path.isfile(file_name):
            print("File: %05d" % i)
            t, v, hdr = rw(file_name, nhdr)
            t1, t2, charge = calculate_charge(t, v, r)
            amplitude = calculate_amp(t, v)
            fwhm = calculate_fwhm(t, v, half_max)
            rise1090, rise2080 = rise_time(t, v, r)
            fall1090, fall2080 = fall_time(t, v, r)
            t1_array = np.append(t1_array, t1)
            t2_array = np.append(t2_array, t2)
            charge_array = np.append(charge_array, charge)
            amplitude_array = np.append(amplitude_array, amplitude)
            fwhm_array = np.append(fwhm_array, fwhm)
            rise1090_array = np.append(rise1090_array, rise1090)
            rise2080_array = np.append(rise2080_array, rise2080)
            fall1090_array = np.append(fall1090_array, fall1090)
            fall2080_array = np.append(fall2080_array, fall2080)

    return t1_array, t2_array, charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, \
           fall1090_array, fall2080_array


def plot_histograms(charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, fall1090_array,
                    fall2080_array):
    plt.hist(fwhm_array, 50)
    plt.xlabel('Time (s)')
    plt.title('FWHM')
    plt.show()

    plt.hist(charge_array, 50)
    plt.xlabel('Charge (C)')
    plt.title('Charge of SPE')
    plt.show()

    plt.hist(amplitude_array, 50)
    plt.xlabel('Voltage (V)')
    plt.title('Amplitude of SPE')
    plt.show()

    plt.hist(rise1090_array, 50)
    plt.xlabel('Time (s)')
    plt.title('10-90 Risetime')
    plt.show()

    plt.hist(rise2080_array, 50)
    plt.xlabel('Time (s)')
    plt.title('20-80 Risetime')
    plt.show()

    plt.hist(fall1090_array, 50)
    plt.xlabel('Time (s)')
    plt.title('10-90 Falltime')
    plt.show()

    plt.hist(fall2080_array, 50)
    plt.xlabel('Time (s)')
    plt.title('20-80 Falltime')
    plt.show()
