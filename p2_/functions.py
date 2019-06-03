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
            print('Reading file #', i)
            t, v, hdr = rw(data_file / file_name, nhdr)     # Reads a waveform file
            v = v / min(v)                                  # Normalizes voltages
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
    # plt.show()

    # Saves average waveform data
    file_name = save_file / (save_name + '.txt')
    hdr = 'Average Waveform\n\n\n\nTime,Ampl'
    ww(t_avg, v_avg, file_name, hdr)


def lowpass_filter(v, tau, fsps):
    alpha = 1 - np.exp(-1. / (fsps * tau))
    v_filtered = np.array([])
    for i in range(len(v)):
        if i == 0:
            v_filtered = np.append(v_filtered, v[i])
        else:
            v2 = v[i] * alpha + (1 - alpha) * v_filtered[i - 1]
            v_filtered = np.append(v_filtered, v2)
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
    idx = np.inf
    idx_min_val = np.where(v == min(v))     # Finds index of minimum voltage value in voltage array
    time_min_val = t[idx_min_val]           # Finds time of point of minimum voltage
    min_time = time_min_val[0]

    tvals = np.linspace(t[0], t[len(t) - 1], 5000)  # Creates array of times over entire timespan
    tvals1 = np.linspace(t[0], min_time, 5000)      # Creates array of times from beginning to point of min voltage
    vvals = np.interp(tvals, t, v)                  # Interpolates & creates array of voltages over entire timespan
    vvals1 = np.interp(tvals1, t, v)   # Interpolates & creates array of voltages from beginning to point of min voltage
    vvals1_flip = np.flip(vvals1)       # Flips array, creating array of voltages from point of min voltage to beginning
    difference_value = vvals1_flip - (0.1 * min(v))    # Finds difference between points in beginning array and 10% max

    for i in range(0, len(difference_value) - 1):  # Starting at point of minimum voltage and going towards beginning
        if difference_value[i] >= 0:               # of waveform, finds where voltage becomes greater than 10% max
            idx = len(difference_value) - i
            break
    if idx == np.inf:       # If voltage never becomes greater than 10% max, finds where voltage is closest to 10% max
        idx = len(difference_value) - 1 - np.argmin(np.abs(difference_value))
    t1 = tvals[np.argmin(np.abs(tvals - tvals1[idx]))]  # Finds time of beginning of spe

    val10 = .1 * (min(v))   # Calculates 10% max
    val90 = 9 * val10       # Calculates 90% max
    tvals2 = np.linspace(t1, min_time, 5000)  # Creates array of times from beginning of spe to point of minimum voltage
    vvals2 = np.interp(tvals2, t, v)     # Interpolates & creates array of voltages from beginning of spe to min voltage

    time10 = tvals[np.argmin(np.abs(vvals2 - val10))]  # Finds time of point of 10% max
    time90 = tvals[np.argmin(np.abs(vvals2 - val90))]  # Finds time of point of 90% max
    rise_time1090 = float(format(time90 - time10, '.2e'))       # Calculates 10-90 rise time

    return rise_time1090
