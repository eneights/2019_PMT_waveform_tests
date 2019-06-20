import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import norm
import random


# Reads csv file with header and time & voltage columns
# Returns time array, voltage array, and header as a string
def rw(file_name, nhdr):
    header = []
    header_str = ''
    x = np.array([])
    y = np.array([])

    if os.path.isfile(file_name):
        myfile = open(file_name, 'rb')          # Opens waveform file
        for i in range(nhdr):                   # Reads header and saves in a list
            header.append(myfile.readline())
        for line in myfile:
            x = np.append(x, float(line.split(str.encode(','))[0]))     # Reads time values & saves in an array
            y = np.append(y, float(line.split(str.encode(','))[1]))     # Reads voltage values & saves in an array
        myfile.close()                          # Closes waveform file
        head_len = len(header)
        for i in range(0, head_len):            # Converts header list to a string
            head_byte = header[i]
            head_str = head_byte.decode('cp437')
            header_str += head_str

    return x, y, header_str


# Given a time array, voltage array, and header, writes a csv file with header and time & voltage columns
def ww(x, y, file_name, hdr):
    myfile = open(file_name, 'w')           # Opens file to write waveform into
    for entry in str(hdr):                  # Writes header to file
        myfile.write(entry)
    for ix, iy in zip(x, y):                # Writes time and voltage values into file
        line = '%.7E,%f\n' % (ix, iy)
        myfile.write(line)
    myfile.close()                          # Closes waveform file


# Given a time array, voltage array, sample rate, and new sample rate, creates downsampled time and voltage arrays
def downsample(t, v, fsps, fsps_new):
    steps = int(fsps / fsps_new + 0.5)
    idx_start = random.randint(0, steps - 1)        # Picks a random index to start at
    t_ds = np.array([])
    v_ds = np.array([])
    for i in range(idx_start, len(v) - 1, steps):   # Creates time & voltage arrays that digitizer would detect
        t_ds = np.append(t_ds, t[i])
        v_ds = np.append(v_ds, v[i])
    return t_ds, v_ds


# Converts voltage array to bits and adds noise
def digitize(v, noise):
    v_bits = np.array([])
    for i in range(len(v)):
        v_bits = np.append(v_bits, (v[i] * (2 ** 14 - 1) * 2 + 0.5))    # Converts voltage array to bits
    noise_array = np.random.normal(scale=noise, size=len(v_bits))        # Creates noise array
    v_digitized = np.add(v_bits, noise_array)        # Adds noise to digitized values
    v_digitized = v_digitized.astype(int)
    return v_digitized


# Returns the average baseline (baseline noise level)
def calculate_average(t, v):
    v_sum = 0

    idx = np.where(v == min(v))     # Finds index of point of minimum voltage value
    idx = idx[0]
    if type(idx) is np.ndarray:
        idx = idx[0]

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


# Returns time when spe waveform begins, time when spe waveform ends, and charge of spe (charge as a positive value)
def calculate_charge(t, v, r):
    vsum = 0
    idx1 = np.inf
    idx2 = np.inf

    min_val = min(v)                        # Finds minimum voltage value
    idx_min_val = np.where(v == min_val)    # Finds index of minimum voltage value in voltage array
    time_min_val = t[idx_min_val]           # Finds time of point of minimum voltage
    min_time = time_min_val[0]

    tvals = np.linspace(t[0], t[len(t) - 1], 5000)      # Creates array of times over entire timespan
    tvals1 = np.linspace(t[0], min_time, 5000)          # Creates array of times from beginning to point of min voltage
    tvals2 = np.linspace(min_time, t[len(t) - 1], 5000)  # Creates array of times from point of min voltage to end
    vvals = np.interp(tvals, t, v)     # Interpolates & creates array of voltages over entire timespan
    vvals1 = np.interp(tvals1, t, v)   # Interpolates & creates array of voltages from beginning to point of min voltage
    vvals2 = np.interp(tvals2, t, v)   # Interpolates & creates array of voltages from point of min voltage to end
    vvals1_flip = np.flip(vvals1)      # Flips array, creating array of voltages from point of min voltage to beginning
    difference_value1 = vvals1_flip - (0.1 * min_val)   # Finds difference between points in beginning array and 10% max
    difference_value2 = vvals2 - (0.1 * min_val)        # Finds difference between points in end array and 10% max

    for i in range(len(difference_value1) - 1):  # Starting at point of minimum voltage and going towards beginning
        if difference_value1[i] >= 0:               # of waveform, finds where voltage becomes greater than 10% max
            idx1 = len(difference_value1) - 1 - i
            break
    if idx1 == np.inf:      # If voltage never becomes greater than 10% max, finds where voltage is closest to 10% max
        idx1 = len(difference_value1) - 1 - np.argmin(np.abs(difference_value1))
    for i in range(len(difference_value2) - 1):  # Starting at point of minimum voltage and going towards end of
        if difference_value2[i] >= 0:               # waveform, finds where voltage becomes greater than 10% max
            idx2 = i
            break
    if idx2 == np.inf:      # If voltage never becomes greater than 10% max, finds where voltage is closest to 10% max
        idx2 = np.argmin(np.abs(difference_value2))

    diff_val1 = np.abs(tvals - tvals1[idx1])        # Finds differences between values in time array and times when
    diff_val2 = np.abs(tvals - tvals2[idx2])        # spe starts and ends
    index1 = np.argmin(diff_val1)       # Finds index of beginning of spe
    index2 = np.argmin(diff_val2)       # Finds index of end of spe
    t1 = tvals[index1]      # Finds time of beginning of spe
    t2 = tvals[index2]      # Finds time of end of spe
    for i in range(len(tvals)):         # Calculates sum of all voltages in full timespan
        vsum += vvals[i]
    charge = -1 * (tvals[len(tvals) - 1]) * vsum / (len(tvals) * r)     # Calculates charge

    return t1, t2, charge


# Returns the amplitude of spe as a positive value (minimum voltage)
def calculate_amp(t, v):
    avg = calculate_average(t, v)       # Calculates value of baseline voltage
    min_val = np.amin(v)                # Calculates minimum voltage
    amp = avg - min_val                 # Calculates max amplitude

    return amp


# Returns the full width half max (FWHM) of spe
def calculate_fwhm(t, v):
    idx_max = -1
    half_max = min(v) / 2                       # Calculates 50% max value
    val_10 = min(v) / 10                        # Calculates 10% max value
    half_max = half_max.item()
    tvals = np.linspace(t[0], t[len(t) - 1], 5000)      # Creates array of times over entire timespan
    vvals = np.interp(tvals, t, v)                      # Interpolates & creates array of voltages over entire timespan
    difference_value = np.abs(vvals - half_max)         # Finds difference between points in voltage array and 50% max
    diff_val = np.abs(vvals - min(v))           # Finds difference between points in voltage array and minimum voltage
    index_min = np.argmin(diff_val)             # Finds index of minimum voltage in voltage array
    differential = np.diff(vvals)               # Finds derivative at every point in voltage array
    for i in range(len(differential) - 1):    # Sets every value in difference_value array with a
        if differential[i] < 0:                                 # negative value equal to infinity
            difference_value[i] = np.inf
    for i in range(index_min.item(), len(vvals - 1)):
        if vvals[i] >= val_10:
            idx_max = i
            break
    try:
        for i in range(idx_max, index_min.item(), -1):
            if vvals[i] <= half_max:
                idx = i
                break
        half_max_time = tvals[idx]          # Finds time of 50% max
        return half_max_time
    except Exception:
        return -1


# Creates text file with time of beginning of spe, time of end of spe, charge, amplitude, fwhm, 10-90 & 20-80 rise
# times, 10-90 & 20-80 fall times, and 10%, 20%, 80% & 90% jitter for an spe file
def save_calculations(file_name, charge, amplitude, fwhm):
    myfile = open(file_name, 'w')
    myfile.write('charge,' + str(charge))
    myfile.write('\namplitude,' + str(amplitude))
    myfile.write('\nfwhm,' + str(fwhm))
    myfile.close()


# Creates text file with data from an array
def write_hist_data(array, dest_path, name):
    array = np.sort(array)
    file_name = Path(dest_path / 'hist_data' / name)

    myfile = open(file_name, 'w')
    for item in array:              # Writes an array item on each line of file
        myfile.write(str(item) + '\n')
    myfile.close()


# Calculates beginning & end times of spe waveform, charge, amplitude, fwhm, 10-90 & 20-80 rise times, 10-90 & 20-80
# fall times, and 10%, 20%, 80% & 90% jitter for each spe file
# Returns arrays of beginning & end times of spe waveform, charge, amplitude, fwhm, 10-90 & 20-80 rise times, 10-90 &
# 20-80 fall times, and 10%, 20%, 80% & 90% jitter
def make_arrays(double_file_array, double_folder, delay_folder, dest_path, nhdr, r, fsps_new):
    charge_array = np.array([])
    amplitude_array = np.array([])
    fwhm_array = np.array([])

    for item in double_file_array:
        file_name1 = str(dest_path / 'double_spe' / double_folder / delay_folder /
                         str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') / 'D3--waveforms--%s.txt') % item
        file_name2 = str(dest_path / 'calculations' / 'double_spe' / double_folder / delay_folder /
                         str(str(int(fsps_new / 1e6)) + '_Msps') / 'D3--waveforms--%s.txt') % item
        if os.path.isfile(file_name1):
            if os.path.isfile(file_name2):      # If the calculations were done previously, they are read from a file
                print("Reading calculations from file #%s" % item)
                myfile = open(file_name2, 'r')      # Opens file with calculations
                csv_reader = csv.reader(myfile)
                file_array = np.array([])
                for row in csv_reader:      # Creates array with calculation data
                    file_array = np.append(file_array, float(row[1]))
                myfile.close()
                charge = file_array[0]
                amplitude = file_array[1]
                fwhm = file_array[2]
                # Any spe waveform that returns impossible values is put into the unusable_data folder
                if charge <= 0 or amplitude <= 0 or fwhm <= 0:
                    raw_file = str(dest_path / double_folder / delay_folder / 'raw' / 'D3--waveforms--%s.txt') % item
                    save_file = str(dest_path / 'unusable_data' / 'D3--waveforms--%s.txt') % item
                    t, v, hdr = rw(raw_file, nhdr)
                    ww(t, v, save_file, hdr)
                    print('Removing file #%s' % item)
                    if os.path.isfile(raw_file):
                        os.remove(raw_file)
                    os.remove(file_name1)
                    os.remove(file_name2)
                    if os.path.isfile(str(dest_path / 'double_spe' / double_folder / delay_folder /
                                          str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                                          'D3--waveforms--%s.txt') % item):
                        os.remove(str(dest_path / 'double_spe' / double_folder / delay_folder /
                                      str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                                      'D3--waveforms--%s.txt') % item)
                # All other double spe waveforms' calculations are placed into arrays
                else:
                    charge_array = np.append(charge_array, charge)
                    amplitude_array = np.append(amplitude_array, amplitude)
                    fwhm_array = np.append(fwhm_array, fwhm)
            else:           # If the calculations were not done yet, they are calculated
                print("Calculating file #%s" % item)
                t, v, hdr = rw(file_name1, nhdr)        # Waveform file is read
                t1, t2, charge = calculate_charge(t, v, r)      # Start & end times and charge are calculated
                amplitude = calculate_amp(t, v)     # Amplitude of spe is calculated
                fwhm = calculate_fwhm(t, v)         # FWHM of spe is calculated
                # Any spe waveform that returns impossible values is put into the unusable_data folder
                if charge <= 0 or amplitude <= 0 or fwhm <= 0:
                    raw_file = str(dest_path / double_folder / delay_folder / 'raw' / 'D3--waveforms--%s.txt') % item
                    save_file = str(dest_path / 'unusable_data' / 'D3--waveforms--%s.txt') % item
                    t, v, hdr = rw(raw_file, nhdr)
                    ww(t, v, save_file, hdr)
                    print('Removing file #%s' % item)
                    if os.path.isfile(raw_file):
                        os.remove(raw_file)
                    os.remove(file_name1)
                    if os.path.isfile(str(dest_path / 'double_spe' / double_folder / delay_folder /
                                          str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                                          'D3--waveforms--%s.txt') % item):
                        os.remove(str(dest_path / 'double_spe' / double_folder / delay_folder /
                                      str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                                      'D3--waveforms--%s.txt') % item)
                # All other double spe waveforms' calculations are saved in a file & placed into arrays
                else:
                    save_calculations(file_name2, charge, amplitude, fwhm)
                    charge_array = np.append(charge_array, charge)
                    amplitude_array = np.append(amplitude_array, amplitude)
                    fwhm_array = np.append(fwhm_array, fwhm)

    return charge_array, amplitude_array, fwhm_array


# Calculates charge, amplitude, and fwhm for each spe file
# Returns arrays of charge, amplitude, and fwhm
def make_arrays_s(single_file_array, dest_path, single_folder, nhdr, r, fsps_new):
    charge_array = np.array([])
    amplitude_array = np.array([])
    fwhm_array = np.array([])

    for item in single_file_array:
        file_name1 = str(dest_path / 'single_spe' / single_folder / str('digitized_' + str(int(fsps_new / 1e6)) +
                                                                        '_Msps') / 'D3--waveforms--%s.txt') % item
        file_name2 = str(dest_path / 'calculations' / 'single_spe' / single_folder /
                         str(str(int(fsps_new / 1e6)) + '_Msps') / 'D3--waveforms--%s.txt') % item
        if os.path.isfile(file_name1):
            if os.path.isfile(file_name2):      # If the calculations were done previously, they are read from a file
                print("Reading calculations from file #%s" % item)
                myfile = open(file_name2, 'r')      # Opens file with calculations
                csv_reader = csv.reader(myfile)
                file_array = np.array([])
                for row in csv_reader:      # Creates array with calculation data
                    file_array = np.append(file_array, float(row[1]))
                myfile.close()
                charge = file_array[0]
                amplitude = file_array[1]
                fwhm = file_array[2]
                # Any spe waveform that returns impossible values is put into the not_spe folder
                if charge <= 0 or amplitude <= 0 or fwhm <= 0:
                    raw_file = str(dest_path / 'single_spe' / single_folder / 'raw' / 'D3--waveforms--%s.txt') % item
                    save_file = str(dest_path / 'unusable_data' / 'D3--waveforms--%s.txt') % item
                    t, v, hdr = rw(raw_file, nhdr)
                    ww(t, v, save_file, hdr)
                    print('Removing file #%s' % item)
                    if os.path.isfile(raw_file):
                        os.remove(raw_file)
                    os.remove(file_name1)
                    os.remove(file_name2)
                    if os.path.isfile(str(dest_path / 'single_spe' / single_folder /
                                          str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                                          'D3--waveforms--%s.txt') % item):
                        os.remove(str(dest_path / 'single_spe' / single_folder /
                                      str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                                      'D3--waveforms--%s.txt') % item)
                # All other spe waveforms' calculations are placed into arrays
                else:
                    charge_array = np.append(charge_array, charge)
                    amplitude_array = np.append(amplitude_array, amplitude)
                    fwhm_array = np.append(fwhm_array, fwhm)
            else:           # If the calculations were not done yet, they are calculated
                print("Calculating file #%s" % item)
                t, v, hdr = rw(file_name1, nhdr)        # Shifted waveform file is read
                t1, t2, charge = calculate_charge(t, v, r)      # Start & end times and charge of spe are calculated
                amplitude = calculate_amp(t, v)     # Amplitude of spe is calculated
                fwhm = calculate_fwhm(t, v)         # FWHM of spe is calculated
                # Any spe waveform that returns impossible values is put into the not_spe folder
                if charge <= 0 or amplitude <= 0 or fwhm <= 0:
                    raw_file = str(dest_path / 'single_spe' / single_folder / 'raw' / 'D3--waveforms--%s.txt') % item
                    save_file = str(dest_path / 'unusable_data' / 'D3--waveforms--%s.txt') % item
                    t, v, hdr = rw(raw_file, nhdr)
                    ww(t, v, save_file, hdr)
                    print('Removing file #%s' % item)
                    if os.path.isfile(raw_file):
                        os.remove(raw_file)
                    os.remove(file_name1)
                    if os.path.isfile(str(dest_path / 'single_spe' / single_folder /
                                          str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                                          'D3--waveforms--%s.txt') % item):
                        os.remove(str(dest_path / 'single_spe' / single_folder /
                                      str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                                      'D3--waveforms--%s.txt') % item)
                # All other spe waveforms' calculations are saved in a file & placed into arrays
                else:
                    save_calculations(file_name2, charge, amplitude, fwhm)
                    charge_array = np.append(charge_array, charge)
                    amplitude_array = np.append(amplitude_array, amplitude)
                    fwhm_array = np.append(fwhm_array, fwhm)

    return charge_array, amplitude_array, fwhm_array


# Creates histogram given an array
def plot_histogram(array, dest_path, nbins, xaxis, title, units, filename, fsps_new):

    def func(x, a, b, c):           # Defines Gaussian function (a is amplitude, b is mean, c is standard deviation)
        return a * np.exp(-(x - b) ** 2.0 / (2 * c ** 2))

    path = Path(dest_path / 'plots')
    n, bins, patches = plt.hist(array, nbins)       # Plots histogram
    b_est, c_est = norm.fit(array)                  # Calculates mean & standard deviation based on entire array
    if filename == 'amp_double_rt4_3x_rt':
        range_min1 = b_est + (c_est / 2) - c_est        # Calculates lower limit of Gaussian fit (1sigma estimation)
        range_max1 = b_est + (c_est / 2) + c_est        # Calculates lower limit of Gaussian fit (1sigma estimation)
    else:
        range_min1 = b_est - c_est              # Calculates lower limit of Gaussian fit (1sigma estimation)
        range_max1 = b_est + c_est              # Calculates upper limit of Gaussian fit (1sigma estimation)

    bins = np.delete(bins, len(bins) - 1)
    bins_diff = bins[1] - bins[0]
    bins = np.linspace(bins[0] + bins_diff / 2, bins[len(bins) - 1] + bins_diff / 2, len(bins))
    bins_range1 = np.linspace(range_min1, range_max1, 10000)  # Creates array of bins between upper & lower limits
    n_range1 = np.interp(bins_range1, bins, n)  # Interpolates & creates array of y axis values
    guess1 = [1, float(b_est), float(c_est)]  # Defines guess for values of a, b & c in Gaussian fit
    popt1, pcov1 = curve_fit(func, bins_range1, n_range1, p0=guess1, maxfev=10000)  # Finds Gaussian fit
    mu1 = float(format(popt1[1], '.2e'))        # Calculates mean based on 1sigma guess
    sigma1 = np.abs(float(format(popt1[2], '.2e')))     # Calculates standard deviation based on 1sigma estimation
    range_min2 = mu1 - 2 * sigma1       # Calculates lower limit of Gaussian fit (2sigma)
    range_max2 = mu1 + 2 * sigma1       # Calculates upper limit of Gaussian fit (2sigma)
    bins_range2 = np.linspace(range_min2, range_max2, 10000)    # Creates array of bins between upper & lower limits
    n_range2 = np.interp(bins_range2, bins, n)      # Interpolates & creates array of y axis values
    guess2 = [1, mu1, sigma1]       # Defines guess for values of a, b & c in Gaussian fit
    popt2, pcov2 = curve_fit(func, bins_range2, n_range2, p0=guess2, maxfev=10000)  # Finds Gaussian fit
    plt.plot(bins_range2, func(bins_range2, *popt2), color='red')       # Plots Gaussian fit (mean +/- 2sigma)
    mu2 = float(format(popt2[1], '.2e'))                # Calculates mean
    sigma2 = np.abs(float(format(popt2[2], '.2e')))     # Calculates standard deviation
    plt.xlabel(xaxis + ' (' + units + ')')
    plt.title(title + ' of SPE\n mean: ' + str(mu2) + ' ' + units + ', SD: ' + str(sigma2) + ' ' + units)
    plt.savefig(path / str(filename + '_' + str(int(fsps_new / 1e6)) + '_Msps' + '.png'), dpi=360)
    plt.close()

    write_hist_data(array, dest_path, filename + '.txt')


def plot_double_hist(dest_path, nbins, xaxis, title, units, filename1, filename2, filename, fsps_new):

    def func(x, a, b, c):           # Defines Gaussian function (a is amplitude, b is mean, c is standard deviation)
        return a * np.exp(-(x - b) ** 2.0 / (2 * c ** 2))

    path = Path(dest_path / 'plots')
    array1 = np.array([])
    array2 = np.array([])

    myfile1 = open(dest_path / 'hist_data' / str(filename1 + '.txt'), 'r')           # Opens histogram 1 file
    for line in myfile1:
        line = line.strip()
        line = float(line)
        array1 = np.append(array1, line)    # Reads values & saves in an array
    myfile1.close()                         # Closes histogram 1 file

    n1, bins1, patches1 = plt.hist(array1, nbins, alpha=0.7)            # Plots histogram 1
    b_est1, c_est1 = norm.fit(array1)           # Calculates mean & standard deviation based on entire array 1

    range_min1_1 = b_est1 - c_est1      # Calculates lower limit of Gaussian fit (1sigma estimation)
    range_max1_1 = b_est1 + c_est1      # Calculates upper limit of Gaussian fit (1sigma estimation)

    bins1 = np.delete(bins1, len(bins1) - 1)
    bins_diff1 = bins1[1] - bins1[0]
    bins1 = np.linspace(bins1[0] + bins_diff1 / 2, bins1[len(bins1) - 1] + bins_diff1 / 2, len(bins1))
    bins_range1_1 = np.linspace(range_min1_1, range_max1_1, 10000)  # Creates array of bins between upper & lower limits
    n_range1_1 = np.interp(bins_range1_1, bins1, n1)        # Interpolates & creates array of y axis values
    guess1_1 = [1, float(b_est1), float(c_est1)]            # Defines guess for values of a, b & c in Gaussian fit
    popt1_1, pcov1_1 = curve_fit(func, bins_range1_1, n_range1_1, p0=guess1_1, maxfev=10000)    # Finds Gaussian fit
    mu1_1 = float(format(popt1_1[1], '.2e'))                # Calculates mean based on 1sigma guess
    sigma1_1 = np.abs(float(format(popt1_1[2], '.2e')))     # Calculates standard deviation based on 1sigma estimation
    range_min2_1 = mu1_1 - 2 * sigma1_1                     # Calculates lower limit of Gaussian fit (2sigma)
    range_max2_1 = mu1_1 + 2 * sigma1_1                     # Calculates upper limit of Gaussian fit (2sigma)
    bins_range2_1 = np.linspace(range_min2_1, range_max2_1, 10000)  # Creates array of bins between upper & lower limits
    n_range2_1 = np.interp(bins_range2_1, bins1, n1)        # Interpolates & creates array of y axis values
    guess2_1 = [1, mu1_1, sigma1_1]                         # Defines guess for values of a, b & c in Gaussian fit
    popt2_1, pcov2_1 = curve_fit(func, bins_range2_1, n_range2_1, p0=guess2_1, maxfev=10000)  # Finds Gaussian fit
    plt.plot(bins_range2_1, func(bins_range2_1, *popt2_1), color='red')     # Plots Gaussian fit (mean +/- 2sigma)
    mu2_1 = float(format(popt2_1[1], '.2e'))                # Calculates mean
    sigma2_1 = np.abs(float(format(popt2_1[2], '.2e')))     # Calculates standard deviation

    myfile2 = open(dest_path / 'hist_data' / str(filename2 + '.txt'), 'r')          # Opens histogram 2 file
    for line in myfile2:
        line = line.strip()
        line = float(line)
        array2 = np.append(array2, line)    # Reads values & saves in an array
    myfile2.close()                         # Closes histogram 2 file

    n2, bins2, patches2 = plt.hist(array2, nbins, alpha=0.7)        # Plots histogram 2
    b_est2, c_est2 = norm.fit(array2)                   # Calculates mean & standard deviation based on entire array 2

    if filename2 == 'amp_double_rt4_3x_rt':
        range_min1_2 = b_est2 + (c_est2 / 2) - c_est2   # Calculates lower limit of Gaussian fit (1sigma estimation)
        range_max1_2 = b_est2 + (c_est2 / 2) + c_est2   # Calculates lower limit of Gaussian fit (1sigma estimation)
    else:
        range_min1_2 = b_est2 - c_est2              # Calculates lower limit of Gaussian fit (1sigma estimation)
        range_max1_2 = b_est2 + c_est2              # Calculates upper limit of Gaussian fit (1sigma estimation)

    bins2 = np.delete(bins2, len(bins2) - 1)
    bins_diff2 = bins2[1] - bins2[0]
    bins2 = np.linspace(bins2[0] + bins_diff2 / 2, bins2[len(bins2) - 1] + bins_diff2 / 2, len(bins2))
    bins_range1_2 = np.linspace(range_min1_2, range_max1_2, 10000)  # Creates array of bins between upper & lower limits
    n_range1_2 = np.interp(bins_range1_2, bins2, n2)        # Interpolates & creates array of y axis values
    guess1_2 = [1, float(b_est2), float(c_est2)]            # Defines guess for values of a, b & c in Gaussian fit
    popt1_2, pcov1_2 = curve_fit(func, bins_range1_2, n_range1_2, p0=guess1_2, maxfev=10000)  # Finds Gaussian fit
    mu1_2 = float(format(popt1_2[1], '.2e'))                # Calculates mean based on 1sigma guess
    sigma1_2 = np.abs(float(format(popt1_2[2], '.2e')))     # Calculates standard deviation based on 1sigma estimation
    range_min2_2 = mu1_2 - 2 * sigma1_2                     # Calculates lower limit of Gaussian fit (2sigma)
    range_max2_2 = mu1_2 + 2 * sigma1_2                     # Calculates upper limit of Gaussian fit (2sigma)
    bins_range2_2 = np.linspace(range_min2_2, range_max2_2, 10000)  # Creates array of bins between upper & lower limits
    n_range2_2 = np.interp(bins_range2_2, bins2, n2)        # Interpolates & creates array of y axis values
    guess2_2 = [1, mu1_2, sigma1_2]                         # Defines guess for values of a, b & c in Gaussian fit
    popt2_2, pcov2_2 = curve_fit(func, bins_range2_2, n_range2_2, p0=guess2_2, maxfev=10000)  # Finds Gaussian fit
    plt.plot(bins_range2_2, func(bins_range2_2, *popt2_2), color='green')   # Plots Gaussian fit (mean +/- 2sigma)
    mu2_2 = float(format(popt2_2[1], '.2e'))                # Calculates mean
    sigma2_2 = np.abs(float(format(popt2_2[2], '.2e')))     # Calculates standard deviation

    plt.xlabel(xaxis + ' (' + units + ')')
    plt.title(title + ' of SPE\n mean (single): ' + str(mu2_1) + ' ' + units + ', SD (single): ' + str(sigma2_1) + ' ' +
              units + '\n mean (double): ' + str(mu2_2) + ' ' + units + ', SD (double): ' + str(sigma2_2) + ' ' + units,
              fontsize='medium')
    plt.savefig(path / str(filename + '_' + str(int(fsps_new / 1e6)) + '_Msps' + '.png'), dpi=360)
    plt.close()


def read_hist_file(path, filename, fsps_new):
    array = np.array([])

    myfile = open(path / str(filename + '_' + str(int(fsps_new / 1e6)) + '_Msps' + '.txt'), 'r')    # Opens file
    for line in myfile:                         # Reads values & saves in an array
        line = line.strip()
        line = float(line)
        array = np.append(array, line)          # Closes histogram file
    myfile.close()

    return array


def false_spes_vs_delay(start, end, factor, parameter, parameter_title, units, fsps_new, means, mean5, mean1,
                        mean15, mean2, mean25, mean3, sds, sd5, sd1, sd15, sd2, sd25, sd3, dest_path):
    cutoff_array = np.array([])
    spes_as_mpes_array = np.array([])
    mpes_as_spes_array = np.array([])
    mpes_as_spes_5x_array = np.array([])
    mpes_as_spes_1x_array = np.array([])
    mpes_as_spes_15x_array = np.array([])
    mpes_as_spes_2x_array = np.array([])
    mpes_as_spes_25x_array = np.array([])
    mpes_as_spes_3x_array = np.array([])

    for i in range(start, end):
        x = i * factor
        cutoff_array = np.append(cutoff_array, x)
        spes_as_mpes = 100 * (1 + ((1 / 2) * (-2 + math.erfc((x - means) / (sds * math.sqrt(2))))))
        mpes_as_spes_5x = 100 * ((1 / 2) * (2 - math.erfc((x - mean5) / (sd5 * math.sqrt(2)))))
        mpes_as_spes_1x = 100 * ((1 / 2) * (2 - math.erfc((x - mean1) / (sd1 * math.sqrt(2)))))
        mpes_as_spes_15x = 100 * ((1 / 2) * (2 - math.erfc((x - mean15) / (sd15 * math.sqrt(2)))))
        mpes_as_spes_2x = 100 * ((1 / 2) * (2 - math.erfc((x - mean2) / (sd2 * math.sqrt(2)))))
        mpes_as_spes_25x = 100 * ((1 / 2) * (2 - math.erfc((x - mean25) / (sd25 * math.sqrt(2)))))
        mpes_as_spes_3x = 100 * ((1 / 2) * (2 - math.erfc((x - mean3) / (sd3 * math.sqrt(2)))))
        spes_as_mpes_array = np.append(spes_as_mpes_array, spes_as_mpes)
        mpes_as_spes_5x_array = np.append(mpes_as_spes_5x_array, mpes_as_spes_5x)
        mpes_as_spes_1x_array = np.append(mpes_as_spes_1x_array, mpes_as_spes_1x)
        mpes_as_spes_15x_array = np.append(mpes_as_spes_15x_array, mpes_as_spes_15x)
        mpes_as_spes_2x_array = np.append(mpes_as_spes_2x_array, mpes_as_spes_2x)
        mpes_as_spes_25x_array = np.append(mpes_as_spes_25x_array, mpes_as_spes_25x)
        mpes_as_spes_3x_array = np.append(mpes_as_spes_3x_array, mpes_as_spes_3x)

    cutoff_array_2 = np.linspace(start * factor, end * factor, 1000)
    spes_as_mpes_array_2 = np.interp(cutoff_array_2, cutoff_array, spes_as_mpes_array)
    mpes_as_spes_5x_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_5x_array)
    mpes_as_spes_1x_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_1x_array)
    mpes_as_spes_15x_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_15x_array)
    mpes_as_spes_2x_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_2x_array)
    mpes_as_spes_25x_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_25x_array)
    mpes_as_spes_3x_array_2 = np.interp(cutoff_array_2, cutoff_array, mpes_as_spes_3x_array)

    idx = np.argmin(np.abs(spes_as_mpes_array_2 - 1))
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_5x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_1x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_15x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_2x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_25x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_3x_array_2[idx])

    delay_array = np.array([1.52e-9, 3.04e-9, 4.56e-9, 6.08e-9, 7.6e-9, 9.12e-9])
    cutoff = str(float(format(cutoff_array_2[idx], '.2e')))

    plt.scatter(delay_array, mpes_as_spes_array)
    plt.plot(delay_array, mpes_as_spes_array)
    plt.xlim(1.3e-9, 9.4e-9)
    plt.ylim(-5, 100)
    plt.xlabel('Delay (s)')
    plt.ylabel('% MPES Misidentified as SPEs')
    plt.title('False SPEs\n' + parameter_title + ' Cutoff = ' + cutoff + ' (' + units + ') (1% False MPEs)')
    for i in range(len(mpes_as_spes_array) - 1):
        pt = str(float(format(mpes_as_spes_array[i], '.1e')))
        plt.annotate(pt + '%', (delay_array[i], mpes_as_spes_array[i] + 1))
    plt.savefig(dest_path / 'plots' / str('false_spes_delay_' + parameter + '_' + str(int(fsps_new / 1e6)) + '_Msps' +
                                          '.png'), dpi=360)
    plt.close()


def false_spes_mpes(start, end, factor, parameter, parameter_title, units, means, meand, sds, sdd, fsps_new, dest_path):
    cutoff_array = np.array([])
    spes_as_mpes_array = np.array([])
    mpes_as_spes_array = np.array([])

    for i in range(start, end):
        x = i * factor
        cutoff_array = np.append(cutoff_array, x)
        spes_as_mpes = 100 * (1 + ((1 / 2) * (-2 + math.erfc((x - means) / (sds * math.sqrt(2))))))
        mpes_as_spes = 100 * ((1 / 2) * (2 - math.erfc((x - meand) / (sdd * math.sqrt(2)))))
        spes_as_mpes_array = np.append(spes_as_mpes_array, spes_as_mpes)
        mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes)

    cutoff_array_2 = np.linspace(start * factor, end * factor, 1000)
    spes_as_mpes_array_2 = np.interp(cutoff_array_2, cutoff_array, spes_as_mpes_array)
    idx = np.argmin(np.abs(spes_as_mpes_array_2 - 1))
    cutoff = float(format(cutoff_array_2[idx], '.2e'))

    plt.plot(cutoff_array, spes_as_mpes_array)
    plt.ylim(-5, 100)
    plt.hlines(1)
    plt.xlabel(parameter_title + ' Cutoff (' + units + ')')
    plt.ylabel('% SPES Misidentified as MPEs')
    plt.title('False MPEs')
    plt.savefig(dest_path / 'plots' / str('false_mpes_' + parameter + '_' + str(int(fsps_new / 1e6)) + '_Msps' +
                                          '.png'), dpi=360)
    plt.close()

    plt.plot(cutoff_array, mpes_as_spes_array)
    plt.ylim(-5, 100)
    plt.vlines(cutoff)
    plt.xlabel(parameter_title + ' Cutoff (' + units + ')')
    plt.ylabel('% MPES Misidentified as SPEs')
    plt.title('False SPEs')
    plt.savefig(dest_path / 'plots' / str('false_spes_' + parameter + '_' + str(int(fsps_new / 1e6)) + '_Msps' +
                                          '.png'), dpi=360)
    plt.close()
