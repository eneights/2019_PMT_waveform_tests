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


# Calculates the average waveform of an spe
def average_waveform(start, end, data_file, dest_path, nhdr, save_name):
    save_file = Path(dest_path / 'plots')
    tsum = 0
    vsum = 0
    n = 0
    for i in range(start, end + 1):
        file_name = 'D2--waveforms--%05d.txt' % i
        if os.path.isfile(data_file / file_name):
            t, v, hdr = rw(data_file / file_name, nhdr)     # Reads a waveform file
            min_v = min(v)
            if min_v == 0:
                ww(t, v, str(dest_path / 'unusable_data' / 'D2--waveforms--%05d.txt') % i, hdr)
                os.remove(data_file / file_name)
                print('Removing file #%05d because its minimum voltage is 0' % i)
            else:
                v = v / min_v                                   # Normalizes voltages
                idx = np.where(t == 0)                          # Finds index of t = 0 point
                idx = int(idx[0])
                t = np.roll(t, -idx)                            # Rolls time array so that t = 0 point is at index 0
                v = np.roll(v, -idx)                            # Rolls voltage array so that 50% max point is at index 0
                idx2 = np.where(t == min(t))                    # Finds index of point of minimum t
                idx2 = int(idx2[0])
                idx3 = np.where(t == max(t))                    # Finds index of point of maximum t
                idx3 = int(idx3[0])
                # Only averages waveform files that have enough points before t = 0 & after the spe
                if idx2 <= 3430:
                    # Removes points between point of maximum t & chosen minimum t in time & voltage arrays
                    t = np.concatenate((t[:idx3], t[3430:]))
                    v = np.concatenate((v[:idx3], v[3430:]))
                    # Rolls time & voltage arrays so that point of chosen minimum t is at index 0
                    t = np.roll(t, -idx3)
                    v = np.roll(v, -idx3)
                    if len(t) >= 3920:
                        # Removes points after chosen point of maximum t in time & voltage arrays
                        t = t[:3920]
                        v = v[:3920]
                        # Sums time & voltage arrays
                        tsum += t
                        vsum += v
                        n += 1
    # Finds average time & voltage arrays
    t_avg = tsum / n
    v_avg = vsum / n

    # Plots average waveform & saves image
    plt.plot(t_avg, v_avg)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Voltage')
    plt.title('Average Waveform')
    plt.savefig(save_file / (save_name + '.png'), dpi=360)
    plt.close()

    # Saves average waveform data
    file_name = save_file / (save_name + '.txt')
    hdr = 'Average Waveform\n\n\n\nTime,Ampl\n'
    ww(t_avg, v_avg, file_name, hdr)


def lowpass_filter(v, tau, fsps):
    v_filtered = np.array([])
    alpha = 1 - np.exp(-1. / (fsps * tau))
    for i in range(len(v)):
        if i == 0:
            v_filtered = np.append(v_filtered, v[i])
        else:
            v_filtered = np.append(v_filtered, v[i] * alpha + (1 - alpha) * v_filtered[i - 1])
    return v_filtered


# Returns the average baseline (baseline noise level)
def calculate_average(t, v):
    v_sum = 0

    idx = np.where(v == min(v))     # Finds index of point of minimum voltage value
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


# Returns 10-90 and 20-80 rise times
def rise_time(t, v, r):
    t1, t2, charge = calculate_charge(t, v, r)      # Calculates start time of spe
    idx_min_val = np.where(v == min(v))     # Finds index of minimum voltage
    time_min_val = t[idx_min_val]           # Finds time at point of minimum voltage
    min_time = time_min_val[0]

    val10 = .1 * (min(v))           # Calculates 10% max
    val20 = 2 * val10               # Calculates 20% max
    val80 = 8 * val10               # Calculates 80% max
    val90 = 9 * val10               # Calculates 90% max
    tvals = np.linspace(t1, min_time, 5000)   # Creates array of times from beginning of spe to point of minimum voltage
    vvals = np.interp(tvals, t, v)   # Interpolates & creates array of voltages from beginning of spe to minimum voltage
    difference_value10 = np.abs(vvals - val10)      # Calculates difference between values in voltage array and 10% max
    difference_value20 = np.abs(vvals - val20)      # Calculates difference between values in voltage array and 20% max
    difference_value80 = np.abs(vvals - val80)      # Calculates difference between values in voltage array and 80% max
    difference_value90 = np.abs(vvals - val90)      # Calculates difference between values in voltage array and 90% max
    index10 = np.argmin(difference_value10)     # Finds index of point of 10% max
    index20 = np.argmin(difference_value20)     # Finds index of point of 20% max
    index80 = np.argmin(difference_value80)     # Finds index of point of 80% max
    index90 = np.argmin(difference_value90)     # Finds index of point of 90% max
    time10 = tvals[index10]         # Finds time of point of 10% max
    time20 = tvals[index20]         # Finds time of point of 20% max
    time80 = tvals[index80]         # Finds time of point of 80% max
    time90 = tvals[index90]         # Finds time of point of 90% max
    rise_time1090 = time90 - time10     # Calculates 10-90 rise time
    rise_time2080 = time80 - time20     # Calculates 20-80 rise time
    rise_time1090 = float(format(rise_time1090, '.2e'))
    rise_time2080 = float(format(rise_time2080, '.2e'))

    return rise_time1090, rise_time2080


def rise_time_1090(t, v):
    idx_start = np.argmin(np.abs(t + 2.5e-8))
    idx_end = np.argmin(np.abs(t + 5e-9))
    v_sum = 0
    for i in range(idx_start.item(), idx_end.item()):
        v_sum += v[i]
    avg = v_sum / (idx_end - idx_start)
    min_v = min(v)
    if min_v == 0:
        return '0 min'

    else:
        idx = np.inf
        idx_min_val = np.where(v == min_v)     # Finds index of minimum voltage value in voltage array
        time_min_val = t[idx_min_val]           # Finds time of point of minimum voltage
        min_time = time_min_val[0]

        tvals = np.linspace(t[0], t[len(t) - 1], 5000)  # Creates array of times over entire timespan
        tvals1 = np.linspace(t[0], min_time, 5000)      # Creates array of times from beginning to point of min voltage
        vvals1 = np.interp(tvals1, t, v)    # Interpolates & creates array of voltages from beginning to point of min
                                            # voltage
        vvals1_flip = np.flip(vvals1)   # Flips array, creating array of voltages from point of min voltage to beginning
        difference_value = vvals1_flip - (0.1 * (min_v - avg))      # Finds difference between points in beginning array
                                                                    # and 10% max
        for i in range(0, len(difference_value) - 1): # Starting at point of minimum voltage and going towards beginning
            if difference_value[i] >= 0:              # of waveform, finds where voltage becomes greater than 10% max
                idx = len(difference_value) - i
                break
        if idx == np.inf:     # If voltage never becomes greater than 10% max, finds where voltage is closest to 10% max
            idx = len(difference_value) - 1 - np.argmin(np.abs(difference_value))
        t1 = tvals[np.argmin(np.abs(tvals - tvals1[idx]))]      # Finds time of beginning of spe

        val10 = .1 * (min_v - avg)      # Calculates 10% max
        val90 = 9 * val10               # Calculates 90% max
        tvals2 = np.linspace(t1, min_time, 5000)    # Creates array of times from beginning of spe to point of minimum
                                                    # voltage
        vvals2 = np.interp(tvals2, t, v)    # Interpolates & creates array of voltages from beginning of spe to minimum
                                            # voltage

        time10 = tvals2[np.argmin(np.abs(vvals2 - val10))]          # Finds time of point of 10% max
        time90 = tvals2[np.argmin(np.abs(vvals2 - val90))]          # Finds time of point of 90% max
        rise_time1090 = float(format(time90 - time10, '.2e'))       # Calculates 10-90 rise time

        return rise_time1090


# Creates text file with 10-90 rise times for an spe file
def save_calculations(save_path, i, filter_1, filter_2, filter_4, filter_8, filter_2_2, filter_2_2_2):
    file_name = str(save_path / 'D2--waveforms--%05d.txt') % i
    myfile = open(file_name, 'w')
    myfile.write('rise1090 no filter,' + str(filter_1))
    myfile.write('\nrise1090 2x filter,' + str(filter_2))
    myfile.write('\nrise1090 4x filter,' + str(filter_4))
    myfile.write('\nrise1090 8x filter,' + str(filter_8))
    myfile.write('\nrise1090 2x 2x filter,' + str(filter_2_2))
    myfile.write('\nrise1090 2x 2x 2x filter,' + str(filter_2_2_2))
    myfile.close()


# Creates text file with data from an array
def write_hist_data(array, dest_path, name):
    array = np.sort(array)
    file_name = Path(dest_path / 'hist_data' / name)

    myfile = open(file_name, 'w')
    for item in array:          # Writes an array item on each line of file
        myfile.write(str(item) + '\n')
    myfile.close()


# Calculates 10-90 rise time for each filter for each spe file
# Returns arrays of 10-90 rise times
def make_arrays(dest_path, save_path, start, end, nhdr):
    filter_1_array = np.array([])
    filter_2_array = np.array([])
    filter_4_array = np.array([])
    filter_8_array = np.array([])
    filter_2_2_array = np.array([])
    filter_2_2_2_array = np.array([])
    error_array = np.array([])

    for i in range(start, end + 1):
        file_name1 = str(dest_path / 'filter1' / 'D2--waveforms--%05d.txt') % i
        file_name2 = str(dest_path / 'filter2' / 'D2--waveforms--%05d.txt') % i
        file_name4 = str(dest_path / 'filter4' / 'D2--waveforms--%05d.txt') % i
        file_name8 = str(dest_path / 'filter8' / 'D2--waveforms--%05d.txt') % i
        file_name2_2 = str(dest_path / 'filter2_2' / 'D2--waveforms--%05d.txt') % i
        file_name2_2_2 = str(dest_path / 'filter2_2_2' / 'D2--waveforms--%05d.txt') % i
        file_name = str(save_path / 'D2--waveforms--%05d.txt') % i
        if os.path.isfile(file_name1) and os.path.isfile(file_name2) and os.path.isfile(file_name4) and \
                os.path.isfile(file_name8) and os.path.isfile(file_name2_2) and os.path.isfile(file_name2_2_2):
            if os.path.isfile(file_name):      # If the calculations were done previously, they are read from a file
                print("Reading calculations from shifted file #%05d" % i)
                myfile = open(file_name2, 'r')      # Opens file with calculations
                csv_reader = csv.reader(myfile)
                file_array = np.array([])
                for row in csv_reader:      # Creates array with calculation data
                    file_array = np.append(file_array, float(row[1]))
                myfile.close()
                filter_1 = file_array[0]
                filter_2 = file_array[1]
                filter_4 = file_array[2]
                filter_8 = file_array[3]
                filter_2_2 = file_array[4]
                filter_2_2_2 = file_array[5]
                # Any spe waveform that returns impossible values is put into an array
                if filter_1 <= 0 or filter_2 <= 0 or filter_4 <= 0 or filter_8 <= 0 or filter_2_2 <= 0 or filter_2_2_2 \
                        <= 0:
                    error_array = np.append(error_array, i)
                # All other spe waveforms' calculations are placed into arrays
                else:
                    filter_1_array = np.append(filter_1_array, filter_1)
                    filter_2_array = np.append(filter_2_array, filter_2)
                    filter_4_array = np.append(filter_4_array, filter_4)
                    filter_8_array = np.append(filter_8_array, filter_8)
                    filter_2_2_array = np.append(filter_2_2_array, filter_2_2)
                    filter_2_2_2_array = np.append(filter_2_2_2_array, filter_2_2_2)
            else:           # If the calculations were not done yet, they are calculated
                print("Calculating shifted file #%05d" % i)
                t1, v1, hdr1 = rw(file_name1, nhdr)        # Unfiltered waveform file is read
                filter_1 = rise_time_1090(t1, v1)     # 10-90 rise time of spe is calculated
                t2, v2, hdr2 = rw(file_name2, nhdr)  # 2x filtered waveform file is read
                filter_2 = rise_time_1090(t2, v2)  # 10-90 rise time of spe is calculated
                t4, v4, hdr4 = rw(file_name4, nhdr)  # 4x filtered waveform file is read
                filter_4 = rise_time_1090(t4, v4)  # 10-90 rise time of spe is calculated
                t8, v8, hdr8 = rw(file_name8, nhdr)  # 8x filtered waveform file is read
                filter_8 = rise_time_1090(t8, v8)  # 10-90 rise time of spe is calculated
                t2_2, v2_2, hdr2_2 = rw(file_name2_2, nhdr)  # 2x 2x filtered waveform file is read
                filter_2_2 = rise_time_1090(t2_2, v2_2)  # 10-90 rise time of spe is calculated
                t2_2_2, v2_2_2, hdr2_2_2 = rw(file_name2_2_2, nhdr)  # 2x 2x 2x filtered waveform file is read
                filter_2_2_2 = rise_time_1090(t2_2_2, v2_2_2)  # 10-90 rise time of spe is calculated
                # Any spe waveform that returns impossible values is removed
                if isinstance(filter_1, str) or filter_1 <= 0:
                    ww(t1, v1, str(dest_path / 'unusable_data' / 'D2--waveforms--%05d.txt') % i, hdr1)
                    os.remove(file_name1)
                    print('Removing file #%05d because its minimum voltage is 0' % i)
                elif isinstance(filter_2, str) or filter_2 <= 0:
                    ww(t2, v2, str(dest_path / 'unusable_data' / 'D2--waveforms--%05d.txt') % i, hdr2)
                    os.remove(file_name2)
                    print('Removing file #%05d because its minimum voltage is 0' % i)
                elif isinstance(filter_4, str) or filter_4 <= 0:
                    ww(t4, v4, str(dest_path / 'unusable_data' / 'D2--waveforms--%05d.txt') % i, hdr4)
                    os.remove(file_name4)
                    print('Removing file #%05d because its minimum voltage is 0' % i)
                elif isinstance(filter_8, str) or filter_8 <= 0:
                    ww(t4, v4, str(dest_path / 'unusable_data' / 'D2--waveforms--%05d.txt') % i, hdr4)
                    os.remove(file_name4)
                    print('Removing file #%05d because its minimum voltage is 0' % i)
                elif isinstance(filter_2_2, str) or filter_2_2 <= 0:
                    ww(t2_2, v2_2, str(dest_path / 'unusable_data' / 'D2--waveforms--%05d.txt') % i, hdr2_2)
                    os.remove(file_name2_2)
                    print('Removing file #%05d because its minimum voltage is 0' % i)
                elif isinstance(filter_2_2_2, str) or filter_2_2_2 <= 0:
                    ww(t2_2_2, v2_2_2, str(dest_path / 'unusable_data' / 'D2--waveforms--%05d.txt') % i, hdr2_2_2)
                    os.remove(file_name2_2_2)
                    print('Removing file #%05d because its minimum voltage is 0' % i)
                # All other spe waveforms' calculations are saved in a file & placed into arrays
                else:
                    save_calculations(save_path, i, filter_1, filter_2, filter_4, filter_8, filter_2_2, filter_2_2_2)
                    filter_1_array = np.append(filter_1_array, filter_1)
                    filter_2_array = np.append(filter_2_array, filter_2)
                    filter_4_array = np.append(filter_4_array, filter_4)
                    filter_8_array = np.append(filter_8_array, filter_8)
                    filter_2_2_array = np.append(filter_2_2_array, filter_2_2)
                    filter_2_2_2_array = np.append(filter_2_2_2_array, filter_2_2_2)

    return filter_1_array, filter_2_array, filter_4_array, filter_8_array, filter_2_2_array, filter_2_2_2_array


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
