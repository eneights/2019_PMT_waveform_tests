import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from pathlib import Path
from scipy import signal
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


def initial_time_arrays(data_shift, start, end, nhdr, r):
    time10_array = np.array([])
    time20_array = np.array([])
    time80_array = np.array([])
    time90_array = np.array([])
    half_time_array = np.array([])

    for i in range(start, end + 1):
        file_name = str(data_shift / 'D1--waveforms--%05d.txt') % i
        if os.path.isfile(file_name):
            print("Calculating file #%05d" % i)
            time10, time20, time80, time90, half_time = calculate_times(file_name, nhdr, r)
            time10_array = np.append(time10_array, time10)
            time20_array = np.append(time20_array, time20)
            time80_array = np.append(time80_array, time80)
            time90_array = np.append(time90_array, time90)
            half_time_array = np.append(half_time_array, half_time)
    return time10_array, time20_array, time80_array, time90_array, half_time_array


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
    time10_2_array = np.array([])
    time20_2_array = np.array([])
    time80_2_array = np.array([])
    time90_2_array = np.array([])
    half_time_2_array = np.array([])

    for i in range(start, end + 1):
        file_name = str(save_shift / 'D1--waveforms--%05d.txt') % i
        if os.path.isfile(file_name):
            print("Calculating shifted file #%05d" % i)
            t, v, hdr = rw(file_name, nhdr)
            t1, t2, charge = calculate_charge(t, v, r)
            amplitude = calculate_amp(t, v)
            fwhm = calculate_fwhm(t, v, half_max)
            rise1090, rise2080 = rise_time(t, v, r)
            fall1090, fall2080 = fall_time(t, v, r)
            time10_shift, time20_shift, time80_shift, time90_shift, half_time_shift = calculate_times(file_name, nhdr,
                                                                                                      r)
            t1_array = np.append(t1_array, t1)
            t2_array = np.append(t2_array, t2)
            charge_array = np.append(charge_array, charge)
            amplitude_array = np.append(amplitude_array, amplitude)
            fwhm_array = np.append(fwhm_array, fwhm)
            rise1090_array = np.append(rise1090_array, rise1090)
            rise2080_array = np.append(rise2080_array, rise2080)
            fall1090_array = np.append(fall1090_array, fall1090)
            fall2080_array = np.append(fall2080_array, fall2080)
            time10_2_array = np.append(time10_2_array, time10_shift)
            time20_2_array = np.append(time20_2_array, time20_shift)
            time80_2_array = np.append(time80_2_array, time80_shift)
            time90_2_array = np.append(time90_2_array, time90_shift)
            half_time_2_array = np.append(half_time_2_array, half_time_shift)

    return t1_array, t2_array, charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, \
           fall1090_array, fall2080_array, time10_2_array, time80_2_array, time90_2_array, half_time_2_array


def plot_histograms(charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, fall1090_array,
                    fall2080_array, time10_array, time20_array, time80_array, time90_array, half_time_array,
                    time10_2_array, time80_2_array, time90_2_array, half_time_2_array):
    plt.hist(fwhm_array, 100, density=True, log=True)
    mu_fwhm, sigma_fwhm = norm.fit(fwhm_array)
    mu_fwhm = float(format(mu_fwhm, '.2e'))
    sigma_fwhm = float(format(sigma_fwhm, '.2e'))
    fwhm_min, fwhm_max = plt.xlim()
    fwhm_x = np.linspace(fwhm_min, fwhm_max, 100)
    fwhm_y = norm.pdf(fwhm_x, mu_fwhm, sigma_fwhm)
    plt.plot(fwhm_x, fwhm_y)
    plt.xlabel('Time (s)')
    plt.title('FWHM\n mean: ' + str(mu_fwhm) + ', SD: ' + str(sigma_fwhm))
    plt.show()

    plt.hist(charge_array, 100, density=True)
    mu_charge, sigma_charge = norm.fit(charge_array)
    mu_charge = float(format(mu_charge, '.2e'))
    sigma_charge = float(format(sigma_charge, '.2e'))
    charge_min, charge_max = plt.xlim()
    charge_x = np.linspace(charge_min, charge_max, 100)
    charge_y = norm.pdf(charge_x, mu_charge, sigma_charge)
    plt.plot(charge_x, charge_y)
    plt.xlabel('Charge (C)')
    plt.title('Charge of SPE\n mean: ' + str(mu_charge) + ', SD: ' + str(sigma_charge))
    plt.show()

    plt.hist(amplitude_array, 100, density=True)
    mu_amplitude, sigma_amplitude = norm.fit(amplitude_array)
    mu_amplitude = float(format(mu_amplitude, '.2e'))
    sigma_amplitude = float(format(sigma_amplitude, '.2e'))
    amplitude_min, amplitude_max = plt.xlim()
    amplitude_x = np.linspace(amplitude_min, amplitude_max, 100)
    amplitude_y = norm.pdf(amplitude_x, mu_amplitude, sigma_amplitude)
    plt.plot(amplitude_x, amplitude_y)
    plt.xlabel('Voltage (V)')
    plt.title('Amplitude of SPE\n mean: ' + str(mu_amplitude) + ', SD: ' + str(sigma_amplitude))
    plt.show()

    plt.hist(rise1090_array, 100, density=True, log=True)
    mu_rise1090, sigma_rise1090 = norm.fit(rise1090_array)
    mu_rise1090 = float(format(mu_rise1090, '.2e'))
    sigma_rise1090 = float(format(sigma_rise1090, '.2e'))
    rise1090_min, rise1090_max = plt.xlim()
    rise1090_x = np.linspace(rise1090_min, rise1090_max, 100)
    rise1090_y = norm.pdf(rise1090_x, mu_rise1090, sigma_rise1090)
    plt.plot(rise1090_x, rise1090_y)
    plt.xlabel('Time (s)')
    plt.title('10-90 Risetime\n mean: ' + str(mu_rise1090) + ', SD: ' + str(sigma_rise1090))
    plt.show()

    plt.hist(rise2080_array, 100, density=True, log=True)
    mu_rise2080, sigma_rise2080 = norm.fit(rise2080_array)
    mu_rise2080 = float(format(mu_rise2080, '.2e'))
    sigma_rise2080 = float(format(sigma_rise2080, '.2e'))
    rise2080_min, rise2080_max = plt.xlim()
    rise2080_x = np.linspace(rise2080_min, rise2080_max, 100)
    rise2080_y = norm.pdf(rise2080_x, mu_rise2080, sigma_rise2080)
    plt.plot(rise2080_x, rise2080_y)
    plt.xlabel('Time (s)')
    plt.title('20-80 Risetime\n mean: ' + str(mu_rise2080) + ', SD: ' + str(sigma_rise2080))
    plt.show()

    plt.hist(fall1090_array, 100, density=True, log=True)
    mu_fall1090, sigma_fall1090 = norm.fit(fall1090_array)
    mu_fall1090 = float(format(mu_fall1090, '.2e'))
    sigma_fall1090 = float(format(sigma_fall1090, '.2e'))
    fall1090_min, fall1090_max = plt.xlim()
    fall1090_x = np.linspace(fall1090_min, fall1090_max, 100)
    fall1090_y = norm.pdf(fall1090_x, mu_fall1090, sigma_fall1090)
    plt.plot(fall1090_x, fall1090_y)
    plt.xlabel('Time (s)')
    plt.title('10-90 Falltime\n mean: ' + str(mu_fall1090) + ', SD: ' + str(sigma_fall1090))
    plt.show()

    plt.hist(fall2080_array, 100, density=True, log=True)
    mu_fall2080, sigma_fall2080 = norm.fit(fall2080_array)
    mu_fall2080 = float(format(mu_fall2080, '.2e'))
    sigma_fall2080 = float(format(sigma_fall2080, '.2e'))
    fall2080_min, fall2080_max = plt.xlim()
    fall2080_x = np.linspace(fall2080_min, fall2080_max, 100)
    fall2080_y = norm.pdf(fall2080_x, mu_fall2080, sigma_fall2080)
    plt.plot(fall2080_x, fall2080_y)
    plt.xlabel('Time (s)')
    plt.title('20-80 Falltime\n mean: ' + str(mu_fall2080) + ', SD: ' + str(sigma_fall2080))
    plt.show()

    plt.hist(time10_array, 100, density=True, log=True)
    mu_time10, sigma_time10 = norm.fit(time10_array)
    mu_time10 = float(format(mu_time10, '.2e'))
    sigma_time10 = float(format(sigma_time10, '.2e'))
    time10_min, time10_max = plt.xlim()
    time10_x = np.linspace(time10_min, time10_max, 100)
    time10_y = norm.pdf(time10_x, mu_time10, sigma_time10)
    plt.plot(time10_x, time10_y)
    plt.xlabel('Time (s)')
    plt.title('Time of 10% Max (before shift)\n mean: ' + str(mu_time10) + ', SD: ' + str(sigma_time10))
    plt.show()

    plt.hist(time20_array, 100, density=True, log=True)
    mu_time20, sigma_time20 = norm.fit(time20_array)
    mu_time20 = float(format(mu_time20, '.2e'))
    sigma_time20 = float(format(sigma_time20, '.2e'))
    time20_min, time20_max = plt.xlim()
    time20_x = np.linspace(time20_min, time20_max, 100)
    time20_y = norm.pdf(time20_x, mu_time20, sigma_time20)
    plt.plot(time20_x, time20_y)
    plt.xlabel('Time (s)')
    plt.title('Time of 20% Max (before shift)\n mean: ' + str(mu_time20) + ', SD: ' + str(sigma_time20))
    plt.show()

    plt.hist(time80_array, 100, density=True, log=True)
    mu_time80, sigma_time80 = norm.fit(time80_array)
    mu_time80 = float(format(mu_time80, '.2e'))
    sigma_time80 = float(format(sigma_time80, '.2e'))
    time80_min, time80_max = plt.xlim()
    time80_x = np.linspace(time80_min, time80_max, 100)
    time80_y = norm.pdf(time80_x, mu_time80, sigma_time80)
    plt.plot(time80_x, time80_y)
    plt.xlabel('Time (s)')
    plt.title('Time of 80% Max (before shift)\n mean: ' + str(mu_time80) + ', SD: ' + str(sigma_time80))
    plt.show()

    plt.hist(time90_array, 100, density=True, log=True)
    mu_time90, sigma_time90 = norm.fit(time90_array)
    mu_time90 = float(format(mu_time90, '.2e'))
    sigma_time90 = float(format(sigma_time90, '.2e'))
    time90_min, time90_max = plt.xlim()
    time90_x = np.linspace(time90_min, time90_max, 100)
    time90_y = norm.pdf(time90_x, mu_time90, sigma_time90)
    plt.plot(time90_x, time90_y)
    plt.xlabel('Time (s)')
    plt.title('Time of 90% Max (before shift)\n mean: ' + str(mu_time90) + ', SD: ' + str(sigma_time90))
    plt.show()

    plt.hist(half_time_array, 100, density=True, log=True)
    mu_half_time, sigma_half_time = norm.fit(half_time_array)
    mu_half_time = float(format(mu_half_time, '.2e'))
    sigma_half_time = float(format(sigma_half_time, '.2e'))
    half_time_min, half_time_max = plt.xlim()
    half_time_x = np.linspace(half_time_min, half_time_max, 100)
    half_time_y = norm.pdf(half_time_x, mu_half_time, sigma_half_time)
    plt.plot(half_time_x, half_time_y)
    plt.xlabel('Time (s)')
    plt.title('Time of Half Max (before shift)\n mean: ' + str(mu_half_time) + ', SD: ' + str(sigma_half_time))
    plt.show()

    plt.hist(time10_2_array, 100, density=True, log=True)
    mu_time10_2, sigma_time10_2 = norm.fit(time10_2_array)
    mu_time10_2 = float(format(mu_time10_2, '.2e'))
    sigma_time10_2 = float(format(sigma_time10_2, '.2e'))
    time10_2_min, time10_2_max = plt.xlim()
    time10_2_x = np.linspace(time10_2_min, time10_2_max, 100)
    time10_2_y = norm.pdf(time10_2_x, mu_time10_2, sigma_time10_2)
    plt.plot(time10_2_x, time10_2_y)
    plt.xlabel('Time (s)')
    plt.title('Time of 10% Max (after shift)\n mean: ' + str(mu_time10_2) + ', SD: ' + str(sigma_time10_2))
    plt.show()

    plt.hist(time20_2_array, 100, density=True, log=True)
    mu_time20_2, sigma_time20_2 = norm.fit(time20_2_array)
    mu_time20_2 = float(format(mu_time20_2, '.2e'))
    sigma_time20_2 = float(format(sigma_time20_2, '.2e'))
    time20_2_min, time20_2_max = plt.xlim()
    time20_2_x = np.linspace(time20_2_min, time20_2_max, 100)
    time20_2_y = norm.pdf(time20_2_x, mu_time20_2, sigma_time20_2)
    plt.plot(time20_2_x, time20_2_y)
    plt.xlabel('Time (s)')
    plt.title('Time of 20% Max (after shift)\n mean: ' + str(mu_time20_2) + ', SD: ' + str(sigma_time20_2))
    plt.show()

    plt.hist(time80_2_array, 100, density=True, log=True)
    mu_time80_2, sigma_time80_2 = norm.fit(time80_2_array)
    mu_time80_2 = float(format(mu_time80_2, '.2e'))
    sigma_time80_2 = float(format(sigma_time80_2, '.2e'))
    time80_2_min, time80_2_max = plt.xlim()
    time80_2_x = np.linspace(time80_2_min, time80_2_max, 100)
    time80_2_y = norm.pdf(time80_2_x, mu_time80_2, sigma_time80_2)
    plt.plot(time80_2_x, time80_2_y)
    plt.xlabel('Time (s)')
    plt.title('Time of 80% Max (after shift)\n mean: ' + str(mu_time80_2) + ', SD: ' + str(sigma_time80_2))
    plt.show()

    plt.hist(time90_2_array, 100, density=True, log=True)
    mu_time90_2, sigma_time90_2 = norm.fit(time90_2_array)
    mu_time90_2 = float(format(mu_time90_2, '.2e'))
    sigma_time90_2 = float(format(sigma_time90_2, '.2e'))
    time90_2_min, time90_2_max = plt.xlim()
    time90_2_x = np.linspace(time90_2_min, time90_2_max, 100)
    time90_2_y = norm.pdf(time90_2_x, mu_time90_2, sigma_time90_2)
    plt.plot(time90_2_x, time90_2_y)
    plt.xlabel('Time (s)')
    plt.title('Time of 90% Max (after shift)\n mean: ' + str(mu_time90_2) + ', SD: ' + str(sigma_time90_2))
    plt.show()

    plt.hist(half_time_2_array, 100, density=True, log=True)
    mu_half_time_2, sigma_half_time_2 = norm.fit(half_time_2_array)
    mu_half_time_2 = float(format(mu_half_time_2, '.2e'))
    sigma_half_time_2 = float(format(sigma_half_time_2, '.2e'))
    half_time_2_min, half_time_2_array_max = plt.xlim()
    half_time_2_x = np.linspace(half_time_2_min, half_time_2_max, 100)
    half_time_2_y = norm.pdf(half_time_2_x, mu_half_time_2, sigma_half_time_2)
    plt.plot(half_time_2_x, half_time_2_y)
    plt.xlabel('Time (s)')
    plt.title('Time of Half Max (after shift)\n mean: ' + str(mu_half_time_2) + ', SD: ' + str(sigma_half_time_2))
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
        half_max = 5 * val10
        tvals = np.linspace(min_time, t2, int(2e6))
        vvals = np.interp(tvals, t, v)
        difference_value10 = np.abs(vvals - val10)
        difference_value20 = np.abs(vvals - val20)
        difference_value80 = np.abs(vvals - val80)
        difference_value90 = np.abs(vvals - val90)
        difference_value_half_max = np.abs(vvals - half_max)
        index10 = np.argmin(difference_value10)
        index20 = np.argmin(difference_value20)
        index80 = np.argmin(difference_value80)
        index90 = np.argmin(difference_value90)
        idx_half_max = np.argmin(difference_value_half_max)
        time10 = tvals[index10]
        time20 = tvals[index20]
        time80 = tvals[index80]
        time90 = tvals[index90]
        half_time = tvals[idx_half_max]

    return time10, time20, time80, time90, half_time
