import os
import csv
import numpy as np
import matplotlib.pyplot as plt
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


# Calculates the average waveform of an spe
def average_waveform(start, end, data_file, dest_path, nhdr, save_name):
    save_file = Path(dest_path / 'plots')
    tsum = 0
    vsum = 0
    n = 0
    for i in range(start, end + 1):
        file_name = 'D3--waveforms--%05d.txt' % i
        if os.path.isfile(data_file / file_name):
            t, v, hdr = rw(data_file / file_name, nhdr)     # Reads a waveform file
            if not min(v) == 0:
                v = v / min(v)                              # Normalizes voltages
                diff_val = np.abs(t)
                idx = int(np.argmin(diff_val))              # Finds index of t = 0 point
                t = np.roll(t, -idx)                        # Rolls time array so that t = 0 point is at index 0
                v = np.roll(v, -idx)                        # Rolls voltage array so that 50% max point is at index 0
                idx2 = np.where(t == min(t))                # Finds index of point of minimum t
                idx2 = int(idx2[0])
                idx3 = np.where(t == max(t))                # Finds index of point of maximum t
                idx3 = int(idx3[0])
                # Only averages waveform files that have enough points before t = 0 & after the spe
                if idx2 <= 86:
                    # Removes points between point of maximum t & chosen minimum t in time & voltage arrays
                    t = np.concatenate((t[:idx3], t[86:]))
                    v = np.concatenate((v[:idx3], v[86:]))
                    # Rolls time & voltage arrays so that point of chosen minimum t is at index 0
                    t = np.roll(t, -idx3)
                    v = np.roll(v, -idx3)
                    if len(t) >= 97:
                        # Removes points after chosen point of maximum t in time & voltage arrays
                        t = t[:97]
                        v = v[:97]
                        # Sums time & voltage arrays
                        tsum += t
                        vsum += v
                        n += 1
    # Finds average time & voltage arrays
    t_avg = tsum / n
    v_avg = vsum / n

    # Plots average waveform & saves image
    plt.scatter(t_avg, v_avg, s=15)
    plt.xlim([min(t_avg) - 8 * (t_avg[1] - t_avg[0]), max(t_avg) + 8 * (t_avg[1] - t_avg[0])])
    plt.plot(t_avg, v_avg)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Voltage')
    plt.title('Average Waveform')
    plt.savefig(save_file / (save_name + '.png'), dpi=360)
    plt.close()

    # Saves average waveform data
    file_name = dest_path / 'hist_data' / (save_name + '.txt')
    hdr = 'Average Waveform\n\n\n\nTime,Ampl\n'
    ww(t_avg, v_avg, file_name, hdr)


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

    for i in range(0, len(difference_value1) - 1):  # Starting at point of minimum voltage and going towards beginning
        if difference_value1[i] >= 0:               # of waveform, finds where voltage becomes greater than 10% max
            idx1 = len(difference_value1) - i
            break
    if idx1 == np.inf:      # If voltage never becomes greater than 10% max, finds where voltage is closest to 10% max
        idx1 = len(difference_value1) - 1 - np.argmin(np.abs(difference_value1))
    for i in range(0, len(difference_value2) - 1):  # Starting at point of minimum voltage and going towards end of
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
    half_max = min(v) / 2       # Calculates 50% max value
    half_max = half_max.item()
    tvals = np.linspace(t[0], t[len(t) - 1], 5000)      # Creates array of times over entire timespan
    vvals = np.interp(tvals, t, v)                      # Interpolates & creates array of voltages over entire timespan
    difference_value = np.abs(vvals - half_max)         # Finds difference between points in voltage array and 50% max
    diff_val = np.abs(vvals - min(v))           # Finds difference between points in voltage array and minimum voltage
    index_min = np.argmin(diff_val)             # Finds index of minimum voltage in voltage array
    differential = np.diff(vvals)               # Finds derivative at every point in voltage array
    for i in range(index_min.item(), len(differential) - 1):    # Sets every value in difference_value array with a
        if differential[i] < 0:                                 # negative value equal to infinity
            difference_value[i] = np.inf
    difference_value = difference_value[index_min.item():len(vvals) - 1]
    idx = np.argmin(difference_value)       # Finds index of 50% max in voltage array
    half_max_time = tvals[idx + index_min.item()]   # Finds time of 50% max

    return half_max_time


# Creates text file with time of beginning of spe, time of end of spe, charge, amplitude, fwhm, 10-90 & 20-80 rise
# times, 10-90 & 20-80 fall times, and 10%, 20%, 80% & 90% jitter for an spe file
def save_calculations(file_name, item, charge, amplitude, fwhm):
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
def make_arrays(double_file_array, double_folder, delay_folder, dest_path, nhdr, r):
    charge_array = np.array([])
    amplitude_array = np.array([])
    fwhm_array = np.array([])

    for item in double_file_array:
        file_name1 = str(dest_path / double_folder / delay_folder / 'digitized' / 'D3--waveforms--%s.txt') % item
        file_name2 = str(dest_path / 'calculations' / double_folder / delay_folder / 'D3--waveforms--%s.txt') % item
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
                    os.remove(raw_file)
                    os.remove(file_name1)
                    os.remove(file_name2)
                    os.remove(str(dest_path / double_folder / delay_folder / 'downsampled' / 'D3--waveforms--%s.txt') %
                              item)
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
                    os.remove(raw_file)
                    os.remove(file_name1)
                    os.remove(file_name2)
                    os.remove(str(dest_path / double_folder / delay_folder / 'downsampled' / 'D3--waveforms--%s.txt') %
                              item)
                # All other double spe waveforms' calculations are saved in a file & placed into arrays
                else:
                    save_calculations(file_name2, item, charge, amplitude, fwhm)
                    charge_array = np.append(charge_array, charge)
                    amplitude_array = np.append(amplitude_array, amplitude)
                    fwhm_array = np.append(fwhm_array, fwhm)

    return charge_array, amplitude_array, fwhm_array


# Creates histogram given an array
def plot_histogram(array, dest_path, nbins, xaxis, title, units, filename):

    def func(x, a, b, c):           # Defines Gaussian function (a is amplitude, b is mean, c is standard deviation)
        return a * np.exp(-(x - b) ** 2.0 / (2 * c ** 2))

    path = Path(dest_path / 'plots')
    n, bins, patches = plt.hist(array, nbins)       # Plots histogram
    b_est, c_est = norm.fit(array)          # Calculates mean & standard deviation based on entire array
    range_min1 = b_est - c_est              # Calculates lower limit of Gaussian fit (1sigma estimation)
    range_max1 = b_est + c_est              # Calculates upper limit of Gaussian fit (1sigma estimation)
    bins = np.delete(bins, len(bins) - 1)
    bins_diff = bins[1] - bins[0]
    bins = np.linspace(bins[0] + bins_diff / 2, bins[len(bins) - 1] + bins_diff / 2, len(bins))
    bins_range1 = np.linspace(range_min1, range_max1, 10000)    # Creates array of bins between upper & lower limits
    n_range1 = np.interp(bins_range1, bins, n)      # Interpolates & creates array of y axis values
    guess1 = [1, float(b_est), float(c_est)]        # Defines guess for values of a, b & c in Gaussian fit
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
    mu2 = float(format(popt2[1], '.2e'))    # Calculates mean
    sigma2 = np.abs(float(format(popt2[2], '.2e')))     # Calculates standard deviation
    plt.xlabel(xaxis + ' (' + units + ')')
    plt.title(title + ' of SPE\n mean: ' + str(mu2) + ' ' + units + ', SD: ' + str(sigma2) + ' ' + units)
    plt.savefig(path / str(filename + '.png'), dpi=360)
    plt.close()

    write_hist_data(array, dest_path, filename + '.txt')
