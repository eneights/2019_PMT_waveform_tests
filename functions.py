import sys
import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from pathlib import Path
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import norm
from read_waveform import read_waveform as rw


def calculate_average(t, v):
    v_sum = 0

    idx_min_val = np.where(v == min(v))
    time_min_val = t[idx_min_val]
    min_time = time_min_val[0]
    idx = np.abs(t - min_time).argmin()

    if idx > len(t) / 2:
        idx1 = int(.1 * len(t))
        idx2 = int(.35 * len(t))
    else:
        idx1 = int(.65 * len(t))
        idx2 = int(.9 * len(t))
    for i in range(idx1, idx2):
        v_sum += v[i]
    average = v_sum / (idx2 - idx1)
    return average


def calculate_charge(t, v, r):
    vsum = 0
    idx1 = np.inf
    idx2 = np.inf

    min_val = min(v)
    idx_min_val = np.where(v == min_val)
    time_min_val = t[idx_min_val]
    min_time = time_min_val[0]

    tvals = np.linspace(t[0], t[len(t) - 1], int(2e6))
    tvals1 = np.linspace(t[0], min_time, int(2e6))
    tvals2 = np.linspace(min_time, t[len(t) - 1], int(2e6))
    vvals = np.interp(tvals, t, v)
    vvals1 = np.interp(tvals1, t, v)
    vvals2 = np.interp(tvals2, t, v)
    vvals1_flip = np.flip(vvals1)
    difference_value1 = vvals1_flip - (0.1 * min_val)
    difference_value2 = vvals2 - (0.1 * min_val)

    for i in range(0, len(difference_value1) - 1):
        if difference_value1[i] >= 0:
            idx1 = len(difference_value1) - i
            break
    if idx1 == np.inf:
        idx1 = len(difference_value1) - 1 - np.argmin(np.abs(difference_value1))
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


def calculate_fwhm(t, v):
    half_max = min(v) / 2
    half_max = half_max.item()
    tvals = np.linspace(t[0], t[len(t) - 1], int(2e6))
    vvals = np.interp(tvals, t, v)
    difference_value = np.abs(vvals - half_max)
    diff_val = np.abs(vvals - min(v))
    index_min = np.argmin(diff_val)
    differential = np.diff(vvals)
    for i in range(index_min.item(), len(differential) - 1):
        if differential[i] < 0:
            difference_value[i] = np.inf
    difference_value = difference_value[index_min.item():len(vvals) - 1]
    idx = np.argmin(difference_value)
    half_max_time = tvals[idx + index_min.item()]
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


def make_arrays(save_shift, dest_path, start, end, nhdr, r):
    t1_array = np.array([])
    t2_array = np.array([])
    charge_array = np.array([])
    amplitude_array = np.array([])
    fwhm_array = np.array([])
    rise1090_array = np.array([])
    rise2080_array = np.array([])
    fall1090_array = np.array([])
    fall2080_array = np.array([])
    time10_array = np.array([])
    time20_array = np.array([])
    time80_array = np.array([])
    time90_array = np.array([])

    for i in range(start, end + 1):
        file_name1 = str(save_shift / 'D1--waveforms--%05d.txt') % i
        file_name2 = str(dest_path / 'calculations' / 'D1--waveforms--%05d.txt') % i
        if os.path.isfile(file_name1):
            if os.path.isfile(file_name2):
                print("Reading calculations from shifted file #%05d" % i)
                myfile = open(file_name2, 'r')
                csv_reader = csv.reader(myfile)
                file_array = np.array([])
                for row in csv_reader:
                    file_array = np.append(file_array, float(row[1]))
                myfile.close()
                t1 = file_array[0]
                t2 = file_array[1]
                charge = file_array[2]
                amplitude = file_array[3]
                fwhm = file_array[4]
                rise1090 = file_array[5]
                rise2080 = file_array[6]
                fall1090 = file_array[7]
                fall2080 = file_array[8]
                time10 = file_array[9]
                time20 = file_array[10]
                time80 = file_array[11]
                time90 = file_array[12]
                t1_array = np.append(t1_array, t1)
                t2_array = np.append(t2_array, t2)
                charge_array = np.append(charge_array, charge)
                amplitude_array = np.append(amplitude_array, amplitude)
                fwhm_array = np.append(fwhm_array, fwhm)
                rise1090_array = np.append(rise1090_array, rise1090)
                rise2080_array = np.append(rise2080_array, rise2080)
                fall1090_array = np.append(fall1090_array, fall1090)
                fall2080_array = np.append(fall2080_array, fall2080)
                time10_array = np.append(time10_array, time10)
                time20_array = np.append(time20_array, time20)
                time80_array = np.append(time80_array, time80)
                time90_array = np.append(time90_array, time90)
            else:
                print("Calculating shifted file #%05d" % i)
                t, v, hdr = rw(file_name1, nhdr)
                t1, t2, charge = calculate_charge(t, v, r)
                amplitude = calculate_amp(t, v)
                fwhm = calculate_fwhm(t, v)
                rise1090, rise2080 = rise_time(t, v, r)
                fall1090, fall2080 = fall_time(t, v, r)
                time10, time20, time80, time90 = calculate_times(file_name1, nhdr, r)
                save_calculations(dest_path, i, t1, t2, charge, amplitude, fwhm, rise1090, rise2080, fall1090, fall2080,
                                  time10, time20, time80, time90)
                t1_array = np.append(t1_array, t1)
                t2_array = np.append(t2_array, t2)
                charge_array = np.append(charge_array, charge)
                amplitude_array = np.append(amplitude_array, amplitude)
                fwhm_array = np.append(fwhm_array, fwhm)
                rise1090_array = np.append(rise1090_array, rise1090)
                rise2080_array = np.append(rise2080_array, rise2080)
                fall1090_array = np.append(fall1090_array, fall1090)
                fall2080_array = np.append(fall2080_array, fall2080)
                time10_array = np.append(time10_array, time10)
                time20_array = np.append(time20_array, time20)
                time80_array = np.append(time80_array, time80)
                time90_array = np.append(time90_array, time90)

    return t1_array, t2_array, charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, \
           fall1090_array, fall2080_array, time10_array, time20_array, time80_array, time90_array


def plot_histograms(charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, fall1090_array,
                    fall2080_array, time10_array, time20_array, time80_array, time90_array, dest_path):

    path = Path(dest_path / 'plots')

    def func(x, a, b, c):
        return a * np.exp(-(x - b) ** 2.0 / (2 * c ** 2))

    n_charge, bins_charge, patches_charge = plt.hist(charge_array, 100)
    bins_charge = np.delete(bins_charge, len(bins_charge) - 1)
    b_est_charge, c_est_charge = norm.fit(charge_array)
    guess_charge = [1, float(b_est_charge), float(c_est_charge)]
    popt, pcov = curve_fit(func, bins_charge, n_charge, p0=guess_charge)
    plt.plot(bins_charge, func(bins_charge, *popt), color='red')
    mu_charge = popt[1]
    sigma_charge = popt[2]
    plt.xlabel('Charge (C)')
    plt.title('Charge of SPE\n mean: ' + str(mu_charge) + ', SD: ' + str(sigma_charge))
    plt.savefig(path / 'charge.png')
    plt.show()

    n_amplitude, bins_amplitude, patches_amplitude = plt.hist(amplitude_array, 100)
    bins_amplitude = np.delete(bins_amplitude, len(bins_amplitude) - 1)
    b_est_amplitude, c_est_amplitude = norm.fit(amplitude_array)
    guess_amplitude = [1, float(b_est_amplitude), float(c_est_amplitude)]
    popt, pcov = curve_fit(func, bins_amplitude, n_amplitude, p0=guess_amplitude)
    plt.plot(bins_amplitude, func(bins_amplitude, *popt), color='red')
    mu_amplitude = popt[1]
    sigma_amplitude = popt[2]
    plt.xlabel('Amplitude (V)')
    plt.title('Amplitude of SPE\n mean: ' + str(mu_amplitude) + ', SD: ' + str(sigma_amplitude))
    plt.savefig(path / 'amplitude.png')
    plt.show()

    n_fwhm, bins_fwhm, patches_fwhm = plt.hist(fwhm_array, 100)
    bins_fwhm = np.delete(bins_fwhm, len(bins_fwhm) - 1)
    b_est_fwhm, c_est_fwhm = norm.fit(fwhm_array)
    guess_fwhm = [1, float(b_est_fwhm), float(c_est_fwhm)]
    popt, pcov = curve_fit(func, bins_fwhm, n_fwhm, p0=guess_fwhm)
    plt.plot(bins_fwhm, func(bins_fwhm, *popt), color='red')
    mu_fwhm = popt[1]
    sigma_fwhm = popt[2]
    plt.xlabel('Time (s)')
    plt.title('FWHM of SPE\n mean: ' + str(mu_fwhm) + ', SD: ' + str(sigma_fwhm))
    plt.savefig(path / 'fwhm.png')
    plt.show()

    n_rise1090, bins_rise1090, patches_rise1090 = plt.hist(rise1090_array, 100)
    bins_rise1090 = np.delete(bins_rise1090, len(bins_rise1090) - 1)
    b_est_rise1090, c_est_rise1090 = norm.fit(rise1090_array)
    guess_rise1090 = [1, float(b_est_rise1090), float(c_est_rise1090)]
    popt, pcov = curve_fit(func, bins_rise1090, n_rise1090, p0=guess_rise1090)
    plt.plot(bins_rise1090, func(bins_rise1090, *popt), color='red')
    mu_rise1090 = popt[1]
    sigma_rise1090 = popt[2]
    plt.xlabel('Time (s)')
    plt.title('10-90 Rise Time of SPE\n mean: ' + str(mu_rise1090) + ', SD: ' + str(sigma_rise1090))
    plt.savefig(path / 'rise1090.png')
    plt.show()

    n_rise2080, bins_rise2080, patches_rise2080 = plt.hist(rise2080_array, 100)
    bins_rise2080 = np.delete(bins_rise2080, len(bins_rise2080) - 1)
    b_est_rise2080, c_est_rise2080 = norm.fit(rise2080_array)
    guess_rise2080 = [1, float(b_est_rise2080), float(c_est_rise2080)]
    popt, pcov = curve_fit(func, bins_rise2080, n_rise2080, p0=guess_rise2080)
    plt.plot(bins_rise2080, func(bins_rise2080, *popt), color='red')
    mu_rise2080 = popt[1]
    sigma_rise2080 = popt[2]
    plt.xlabel('Time (s)')
    plt.title('20-80 Rise Time of SPE\n mean: ' + str(mu_rise2080) + ', SD: ' + str(sigma_rise2080))
    plt.savefig(path / 'rise2080.png')
    plt.show()

    n_fall1090, bins_fall1090, patches_fall1090 = plt.hist(fall1090_array, 100)
    bins_fall1090 = np.delete(bins_fall1090, len(bins_fall1090) - 1)
    b_est_fall1090, c_est_fall1090 = norm.fit(fall1090_array)
    guess_fall1090 = [1, float(b_est_fall1090), float(c_est_fall1090)]
    popt, pcov = curve_fit(func, bins_fall1090, n_fall1090, p0=guess_fall1090)
    plt.plot(bins_fall1090, func(bins_fall1090, *popt), color='red')
    mu_fall1090 = popt[1]
    sigma_fall1090 = popt[2]
    plt.xlabel('Time (s)')
    plt.title('10-90 Fall Time of SPE\n mean: ' + str(mu_fall1090) + ', SD: ' + str(sigma_fall1090))
    plt.savefig(path / 'fall1090.png')
    plt.show()

    n_fall2080, bins_fall2080, patches_fall2080 = plt.hist(fall2080_array, 100)
    bins_fall2080 = np.delete(bins_fall2080, len(bins_fall2080) - 1)
    b_est_fall2080, c_est_fall2080 = norm.fit(fall2080_array)
    guess_fall2080 = [1, float(b_est_fall2080), float(c_est_fall2080)]
    popt, pcov = curve_fit(func, bins_fall2080, n_fall2080, p0=guess_fall2080)
    plt.plot(bins_fall2080, func(bins_fall2080, *popt), color='red')
    mu_fall2080 = popt[1]
    sigma_fall2080 = popt[2]
    plt.xlabel('Time (s)')
    plt.title('20-80 Fall Time of SPE\n mean: ' + str(mu_fall2080) + ', SD: ' + str(sigma_fall2080))
    plt.savefig(path / 'fall2080.png')
    plt.show()

    n_time10, bins_time10, patches_time10 = plt.hist(time10_array, 100)
    bins_time10 = np.delete(bins_time10, len(bins_time10) - 1)
    b_est_time10, c_est_time10 = norm.fit(time10_array)
    guess_time10 = [1, float(b_est_time10), float(c_est_time10)]
    popt, pcov = curve_fit(func, bins_time10, n_time10, p0=guess_time10)
    plt.plot(bins_time10, func(bins_time10, *popt), color='red')
    mu_time10 = popt[1]
    sigma_time10 = popt[2]
    plt.xlabel('Time (s)')
    plt.title('10% Jitter of SPE\n mean: ' + str(mu_time10) + ', SD: ' + str(sigma_time10))
    plt.savefig(path / 'time10.png')
    plt.show()

    n_time20, bins_time20, patches_time20 = plt.hist(time20_array, 100)
    bins_time20 = np.delete(bins_time20, len(bins_time20) - 1)
    b_est_time20, c_est_time20 = norm.fit(time20_array)
    guess_time20 = [1, float(b_est_time20), float(c_est_time20)]
    popt, pcov = curve_fit(func, bins_time20, n_time20, p0=guess_time20)
    plt.plot(bins_time20, func(bins_time20, *popt), color='red')
    mu_time20 = popt[1]
    sigma_time20 = popt[2]
    plt.xlabel('Time (s)')
    plt.title('20% Jitter of SPE\n mean: ' + str(mu_time20) + ', SD: ' + str(sigma_time20))
    plt.savefig(path / 'time20.png')
    plt.show()

    n_time80, bins_time80, patches_time80 = plt.hist(time80_array, 100)
    bins_time80 = np.delete(bins_time80, len(bins_time80) - 1)
    b_est_time80, c_est_time80 = norm.fit(time80_array)
    guess_time80 = [1, float(b_est_time80), float(c_est_time80)]
    popt, pcov = curve_fit(func, bins_time80, n_time80, p0=guess_time80)
    plt.plot(bins_time80, func(bins_time80, *popt), color='red')
    mu_time80 = popt[1]
    sigma_time80 = popt[2]
    plt.xlabel('Time (s)')
    plt.title('80% Jitter of SPE\n mean: ' + str(mu_time80) + ', SD: ' + str(sigma_time80))
    plt.savefig(path / 'time80.png')
    plt.show()

    n_time90, bins_time90, patches_time90 = plt.hist(time90_array, 100)
    bins_time90 = np.delete(bins_time90, len(bins_time90) - 1)
    b_est_time90, c_est_time90 = norm.fit(time90_array)
    guess_time90 = [1, float(b_est_time90), float(c_est_time90)]
    popt, pcov = curve_fit(func, bins_time90, n_time90, p0=guess_time90)
    plt.plot(bins_time90, func(bins_time90, *popt), color='red')
    mu_time90 = popt[1]
    sigma_time90 = popt[2]
    plt.xlabel('Time (s)')
    plt.title('90% Jitter of SPE\n mean: ' + str(mu_time90) + ', SD: ' + str(sigma_time90))
    plt.savefig(path / 'time90.png')
    plt.show()


def calculate_times(file_name, nhdr, r):
    if os.path.isfile(file_name):
        t, v, hdr = rw(file_name, nhdr)
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

        return time10, time20, time80, time90


def remove_outliers(charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, fall1090_array,
                    fall2080_array, time10_array, time20_array, time80_array, time90_array):
    charge_array = np.sort(charge_array)
    charge_med = np.median(charge_array)
    charge_diff = np.abs(charge_array - charge_med)
    charge_idx = np.argmin(charge_diff)
    charge_array1 = charge_array[0:charge_idx]
    charge_array2 = charge_array[charge_idx:len(charge_array) - 1]
    charge_q1 = np.median(charge_array1)
    charge_q3 = np.median(charge_array2)
    charge_iq_range = charge_q3 - charge_q1
    charge_outer1 = charge_q1 - (3 * charge_iq_range)
    charge_outer2 = charge_q3 + (3 * charge_iq_range)
    charge_diff1 = np.abs(charge_array - charge_outer1)
    charge_diff2 = np.abs(charge_array - charge_outer2)
    idx1 = np.argmin(charge_diff1)
    idx2 = np.argmin(charge_diff2)
    charge_array = charge_array[idx1:idx2]

    amplitude_array = np.sort(amplitude_array)
    amplitude_med = np.median(amplitude_array)
    amplitude_diff = np.abs(amplitude_array - amplitude_med)
    amplitude_idx = np.argmin(amplitude_diff)
    amplitude_array1 = amplitude_array[0:amplitude_idx]
    amplitude_array2 = amplitude_array[amplitude_idx:len(amplitude_array) - 1]
    amplitude_q1 = np.median(amplitude_array1)
    amplitude_q3 = np.median(amplitude_array2)
    amplitude_iq_range = amplitude_q3 - amplitude_q1
    amplitude_outer1 = amplitude_q1 - (3 * amplitude_iq_range)
    amplitude_outer2 = amplitude_q3 + (3 * amplitude_iq_range)
    amplitude_diff1 = np.abs(amplitude_array - amplitude_outer1)
    amplitude_diff2 = np.abs(amplitude_array - amplitude_outer2)
    idx1 = np.argmin(amplitude_diff1)
    idx2 = np.argmin(amplitude_diff2)
    amplitude_array = amplitude_array[idx1:idx2]

    fwhm_array = np.sort(fwhm_array)
    fwhm_med = np.median(fwhm_array)
    fwhm_diff = np.abs(fwhm_array - fwhm_med)
    fwhm_idx = np.argmin(fwhm_diff)
    fwhm_array1 = fwhm_array[0:fwhm_idx]
    fwhm_array2 = fwhm_array[fwhm_idx:len(fwhm_array) - 1]
    fwhm_q1 = np.median(fwhm_array1)
    fwhm_q3 = np.median(fwhm_array2)
    fwhm_iq_range = fwhm_q3 - fwhm_q1
    fwhm_outer1 = fwhm_q1 - (3 * fwhm_iq_range)
    fwhm_outer2 = fwhm_q3 + (3 * fwhm_iq_range)
    fwhm_diff1 = np.abs(fwhm_array - fwhm_outer1)
    fwhm_diff2 = np.abs(fwhm_array - fwhm_outer2)
    idx1 = np.argmin(fwhm_diff1)
    idx2 = np.argmin(fwhm_diff2)
    fwhm_array = fwhm_array[idx1:idx2]

    rise1090_array = np.sort(rise1090_array)
    rise1090_med = np.median(rise1090_array)
    rise1090_diff = np.abs(rise1090_array - rise1090_med)
    rise1090_idx = np.argmin(rise1090_diff)
    rise1090_array1 = rise1090_array[0:rise1090_idx]
    rise1090_array2 = rise1090_array[rise1090_idx:len(rise1090_array) - 1]
    rise1090_q1 = np.median(rise1090_array1)
    rise1090_q3 = np.median(rise1090_array2)
    rise1090_iq_range = rise1090_q3 - rise1090_q1
    rise1090_outer1 = rise1090_q1 - (3 * rise1090_iq_range)
    rise1090_outer2 = rise1090_q3 + (3 * rise1090_iq_range)
    rise1090_diff1 = np.abs(rise1090_array - rise1090_outer1)
    rise1090_diff2 = np.abs(rise1090_array - rise1090_outer2)
    idx1 = np.argmin(rise1090_diff1)
    idx2 = np.argmin(rise1090_diff2)
    rise1090_array = rise1090_array[idx1:idx2]

    rise2080_array = np.sort(rise2080_array)
    rise2080_med = np.median(rise2080_array)
    rise2080_diff = np.abs(rise2080_array - rise2080_med)
    rise2080_idx = np.argmin(rise2080_diff)
    rise2080_array1 = rise2080_array[0:rise2080_idx]
    rise2080_array2 = rise2080_array[rise2080_idx:len(rise2080_array) - 1]
    rise2080_q1 = np.median(rise2080_array1)
    rise2080_q3 = np.median(rise2080_array2)
    rise2080_iq_range = rise2080_q3 - rise2080_q1
    rise2080_outer1 = rise2080_q1 - (3 * rise2080_iq_range)
    rise2080_outer2 = rise2080_q3 + (3 * rise2080_iq_range)
    rise2080_diff1 = np.abs(rise2080_array - rise2080_outer1)
    rise2080_diff2 = np.abs(rise2080_array - rise2080_outer2)
    idx1 = np.argmin(rise2080_diff1)
    idx2 = np.argmin(rise2080_diff2)
    rise2080_array = rise2080_array[idx1:idx2]

    fall1090_array = np.sort(fall1090_array)
    fall1090_med = np.median(fall1090_array)
    fall1090_diff = np.abs(fall1090_array - fall1090_med)
    fall1090_idx = np.argmin(fall1090_diff)
    fall1090_array1 = fall1090_array[0:fall1090_idx]
    fall1090_array2 = fall1090_array[fall1090_idx:len(fall1090_array) - 1]
    fall1090_q1 = np.median(fall1090_array1)
    fall1090_q3 = np.median(fall1090_array2)
    fall1090_iq_range = fall1090_q3 - fall1090_q1
    fall1090_outer1 = fall1090_q1 - (3 * fall1090_iq_range)
    fall1090_outer2 = fall1090_q3 + (3 * fall1090_iq_range)
    fall1090_diff1 = np.abs(fall1090_array - fall1090_outer1)
    fall1090_diff2 = np.abs(fall1090_array - fall1090_outer2)
    idx1 = np.argmin(fall1090_diff1)
    idx2 = np.argmin(fall1090_diff2)
    fall1090_array = fall1090_array[idx1:idx2]

    fall2080_array = np.sort(fall2080_array)
    fall2080_med = np.median(fall2080_array)
    fall2080_diff = np.abs(fall2080_array - fall2080_med)
    fall2080_idx = np.argmin(fall2080_diff)
    fall2080_array1 = fall2080_array[0:fall2080_idx]
    fall2080_array2 = fall2080_array[fall2080_idx:len(fall2080_array) - 1]
    fall2080_q1 = np.median(fall2080_array1)
    fall2080_q3 = np.median(fall2080_array2)
    fall2080_iq_range = fall2080_q3 - fall2080_q1
    fall2080_outer1 = fall2080_q1 - (3 * fall2080_iq_range)
    fall2080_outer2 = fall2080_q3 + (3 * fall2080_iq_range)
    fall2080_diff1 = np.abs(fall2080_array - fall2080_outer1)
    fall2080_diff2 = np.abs(fall2080_array - fall2080_outer2)
    idx1 = np.argmin(fall2080_diff1)
    idx2 = np.argmin(fall2080_diff2)
    fall2080_array = fall2080_array[idx1:idx2]

    time10_array = np.sort(time10_array)
    time10_med = np.median(time10_array)
    time10_diff = np.abs(time10_array - time10_med)
    time10_idx = np.argmin(time10_diff)
    time10_array1 = time10_array[0:time10_idx]
    time10_array2 = time10_array[time10_idx:len(time10_array) - 1]
    time10_q1 = np.median(time10_array1)
    time10_q3 = np.median(time10_array2)
    time10_iq_range = time10_q3 - time10_q1
    time10_outer1 = time10_q1 - (3 * time10_iq_range)
    time10_outer2 = time10_q3 + (3 * time10_iq_range)
    time10_diff1 = np.abs(time10_array - time10_outer1)
    time10_diff2 = np.abs(time10_array - time10_outer2)
    idx1 = np.argmin(time10_diff1)
    idx2 = np.argmin(time10_diff2)
    time10_array = time10_array[idx1:idx2]

    time20_array = np.sort(time20_array)
    time20_med = np.median(time20_array)
    time20_diff = np.abs(time20_array - time20_med)
    time20_idx = np.argmin(time20_diff)
    time20_array1 = time20_array[0:time20_idx]
    time20_array2 = time20_array[time20_idx:len(time20_array) - 1]
    time20_q1 = np.median(time20_array1)
    time20_q3 = np.median(time20_array2)
    time20_iq_range = time20_q3 - time20_q1
    time20_outer1 = time20_q1 - (3 * time20_iq_range)
    time20_outer2 = time20_q3 + (3 * time20_iq_range)
    time20_diff1 = np.abs(time20_array - time20_outer1)
    time20_diff2 = np.abs(time20_array - time20_outer2)
    idx1 = np.argmin(time20_diff1)
    idx2 = np.argmin(time20_diff2)
    time20_array = time20_array[idx1:idx2]

    time80_array = np.sort(time80_array)
    time80_med = np.median(time80_array)
    time80_diff = np.abs(time80_array - time80_med)
    time80_idx = np.argmin(time80_diff)
    time80_array1 = time80_array[0:time80_idx]
    time80_array2 = time80_array[time80_idx:len(time80_array) - 1]
    time80_q1 = np.median(time80_array1)
    time80_q3 = np.median(time80_array2)
    time80_iq_range = time80_q3 - time80_q1
    time80_outer1 = time80_q1 - (3 * time80_iq_range)
    time80_outer2 = time80_q3 + (3 * time80_iq_range)
    time80_diff1 = np.abs(time80_array - time80_outer1)
    time80_diff2 = np.abs(time80_array - time80_outer2)
    idx1 = np.argmin(time80_diff1)
    idx2 = np.argmin(time80_diff2)
    time80_array = time80_array[idx1:idx2]

    time90_array = np.sort(time90_array)
    time90_med = np.median(time90_array)
    time90_diff = np.abs(time90_array - time90_med)
    time90_idx = np.argmin(time90_diff)
    time90_array1 = time90_array[0:time90_idx]
    time90_array2 = time90_array[time90_idx:len(time90_array) - 1]
    time90_q1 = np.median(time90_array1)
    time90_q3 = np.median(time90_array2)
    time90_iq_range = time90_q3 - time90_q1
    time90_outer1 = time90_q1 - (3 * time90_iq_range)
    time90_outer2 = time90_q3 + (3 * time90_iq_range)
    time90_diff1 = np.abs(time90_array - time90_outer1)
    time90_diff2 = np.abs(time90_array - time90_outer2)
    idx1 = np.argmin(time90_diff1)
    idx2 = np.argmin(time90_diff2)
    time90_array = time90_array[idx1:idx2]

    return charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, fall1090_array, fall2080_array, \
           time10_array, time20_array, time80_array, time90_array


def save_calculations(dest_path, i, t1, t2, charge, amplitude, fwhm, rise1090, rise2080, fall1090, fall2080, time10,
                      time20, time80, time90):
    file_name = str(dest_path / 'calculations' / 'D1--waveforms--%05d.txt') % i
    myfile = open(file_name, 'w')
    myfile.write('t1,' + str(t1))
    myfile.write('\nt2,' + str(t2))
    myfile.write('\ncharge,' + str(charge))
    myfile.write('\namplitude,' + str(amplitude))
    myfile.write('\nfwhm,' + str(fwhm))
    myfile.write('\nrise1090,' + str(rise1090))
    myfile.write('\nrise2080,' + str(rise2080))
    myfile.write('\nfall1090,' + str(fall1090))
    myfile.write('\nfall2080,' + str(fall2080))
    myfile.write('\ntime10,' + str(time10))
    myfile.write('\ntime20,' + str(time20))
    myfile.write('\ntime80,' + str(time80))
    myfile.write('\ntime90,' + str(time90))
    myfile.close()