import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import norm


# Reads csv file with header and time & voltage columns
# Returns time array, voltage array, and header as a string
def rw(file_name, nhdr):
    header = []
    header_str = ''
    x = np.array([])
    y = np.array([])

    if os.path.isfile(file_name):
        myfile = open(file_name, 'rb')
        for i in range(nhdr):
            header.append(myfile.readline())
        for line in myfile:
            x = np.append(x, float(line.split(str.encode(','))[0]))
            y = np.append(y, float(line.split(str.encode(','))[1]))
        myfile.close()
        head_len = len(header)
        for i in range(0, head_len):
            head_byte = header[i]
            head_str = head_byte.decode('cp437')
            header_str += head_str
    return x, y, header_str


# Given a time array, voltage array, and header, writes a csv file with header and time & voltage columns
def ww(x, y, file_name, hdr):
    myfile = open(file_name, 'w')
    for entry in str(hdr):
        myfile.write(entry)
    for ix, iy in zip(x, y):
        line = '%.7E,%f\n' % (ix, iy)
        myfile.write(line)
    myfile.close()


# Returns the average baseline (baseline noise level)
def calculate_average(t, v):
    v_sum = 0

    idx_min_val = np.where(v == min(v))
    time_min_val = t[idx_min_val]
    min_time = time_min_val[0]
    idx = np.abs(t - min_time).argmin()

    if idx > len(t) / 2:            # If minimum voltage is in second half of voltage array, calculates baseline using
        idx1 = int(.1 * len(t))     # first half of voltage array
        idx2 = int(.35 * len(t))
    else:
        idx1 = int(.65 * len(t))    # If minimum voltage is in first half of voltage array, calculates baseline using
        idx2 = int(.9 * len(t))     # second half of voltage array
    for i in range(idx1, idx2):
        v_sum += v[i]
    average = v_sum / (idx2 - idx1)
    return average


# Shifts spes so that baseline = 0 and when t = 0, v = 50% max
def subtract_time(file_num, nhdr, data_path, save_path):
    file_name = 'D1--waveforms--%05d.txt' % file_num

    if os.path.isfile(data_path / file_name):
        if os.path.isfile(save_path / file_name):
            pass
        else:
            t, v, hdr = rw(data_path / file_name, nhdr)
            half_max = min(v) / 2
            differential = np.diff(v)
            difference_value = np.abs(v - half_max)
            for i in range(0, len(differential)):
                if differential[i] > 0:
                    difference_value[i] = np.inf
            index = np.argmin(difference_value)
            half_max_time = t[index]
            t2 = t - half_max_time
            avg = calculate_average(t, v)
            v2 = v - avg
            ww(t2, v2, save_path / file_name, hdr)
            print('Length of /d1_shifted/:', len(os.listdir(str(save_path))))


# Returns time when spe waveform begins, time when spe waveform ends, and charge of spe (charge as a positive value)
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
    difference_value1 = vvals1_flip - (0.05 * min_val)
    difference_value2 = vvals2 - (0.05 * min_val)

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
    for i in range(len(tvals)):
        vsum += vvals[i]
    charge = -1 * (tvals[len(tvals) - 1]) * vsum / (len(tvals) * r)
    return t1, t2, charge


# Returns the amplitude of spe as a positive value (minimum voltage)
def calculate_amp(t, v):
    avg = calculate_average(t, v)
    min_val = np.amin(v)
    amp = avg - min_val
    return amp


# Returns the full width half max (FWHM) of spe
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


# Returns 10-90 and 20-80 rise times
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


# Returns 10-90 and 20-80 fall times
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


# Returns 10%, 20%, 80%, and 90% jitter of spe
def calculate_times(t, v, r):
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


# Creates text file with time of beginning of spe, time of end of spe, charge, amplitude, fwhm, 10-90 & 20-80 rise
# times, 10-90 & 20-80 fall times, and 10%, 20%, 80% & 90% jitter for an spe file
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


# Creates text file with data from an array
def write_hist_data(array, dest_path, name):
    array = np.sort(array)
    file_name = Path(dest_path / 'hist_data' / name)

    myfile = open(file_name, 'w')
    for item in array:
        myfile.write(str(item) + '\n')
    myfile.close()


# Calculates beginning & end times of spe waveform, charge, amplitude, fwhm, 10-90 & 20-80 rise times, 10-90 & 20-80
# fall times, and 10%, 20%, 80% & 90% jitter for each spe file
# Returns arrays of beginning & end times of spe waveform, charge, amplitude, fwhm, 10-90 & 20-80 rise times, 10-90 &
# 20-80 fall times, and 10%, 20%, 80% & 90% jitter
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
                time10, time20, time80, time90 = calculate_times(t, v, r)
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

    t1_array = np.sort(t1_array)
    t2_array = np.sort(t2_array)
    charge_array = np.sort(charge_array)
    amplitude_array = np.sort(amplitude_array)
    fwhm_array = np.sort(fwhm_array)
    rise1090_array = np.sort(rise1090_array)
    rise2080_array = np.sort(rise2080_array)
    fall1090_array = np.sort(fall1090_array)
    fall2080_array = np.sort(fall2080_array)
    time10_array = np.sort(time10_array)
    time20_array = np.sort(time20_array)
    time80_array = np.sort(time80_array)
    time90_array = np.sort(time90_array)

    return t1_array, t2_array, charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, \
           fall1090_array, fall2080_array, time10_array, time20_array, time80_array, time90_array


# Creates histogram given an array
def plot_histogram(array, dest_path, nbins, xaxis, title, units, filename):

    def func(x, a, b, c):
        return a * np.exp(-(x - b) ** 2.0 / (2 * c ** 2))

    path = Path(dest_path / 'plots')
    n, bins, patches = plt.hist(array, nbins)
    bins = np.delete(bins, len(bins) - 1)
    b_est, c_est = norm.fit(array)
    guess = [1, float(b_est), float(c_est)]
    popt, pcov = curve_fit(func, bins, n, p0=guess, maxfev=10000)
    plt.plot(bins, func(bins, *popt), color='red')
    mu = float(format(popt[1], '.2e'))
    sigma = np.abs(float(format(popt[2], '.2e')))
    plt.xlabel(xaxis + ' (' + units + ')')
    plt.title(title + ' of SPE\n mean: ' + str(mu) + ' ' + units + ', SD: ' + str(sigma) + ' ' + units)
    plt.savefig(path / str(filename + '.png'), dpi=360)
    plt.show()

    write_hist_data(array, dest_path, filename + '.txt')


# Returns array with outliers removed
def remove_outliers(array, nbins, nsigma):

    def func(x, a, b, c):
        return a * np.exp(-(x - b) ** 2.0 / (2 * c ** 2))

    n, bins, patches = plt.hist(array, nbins)
    bins = np.delete(bins, len(bins) - 1)
    b_est, c_est = norm.fit(array)
    guess = [1, float(b_est), float(c_est)]
    popt, pcov = curve_fit(func, bins, n, p0=guess, maxfev=2000)
    mu = float(format(popt[1], '.2e'))
    sigma = float(format(popt[2], '.2e'))
    minimum = mu - nsigma * np.abs(sigma)
    maximum = mu + nsigma * np.abs(sigma)
    min_diff_val = np.abs(array - minimum)
    max_diff_val = np.abs(array - maximum)
    min_idx = np.argmin(min_diff_val)
    max_idx = np.argmin(max_diff_val)
    array = array[min_idx:max_idx]
    plt.close()

    return array
