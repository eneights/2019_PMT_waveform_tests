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


# Returns the average baseline (baseline noise level)
def calculate_average(t, v):
    v_sum = 0

    idx = np.where(v == min(v))     # Finds index of point of minimum voltage value

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
def shift_waveform(file_num, nhdr, data_path, save_path):
    file_name = 'D1--waveforms--%05d.txt' % file_num

    if os.path.isfile(data_path / file_name):
        if os.path.isfile(save_path / file_name):       # If file has already been shifted, does nothing
            pass
        else:
            t, v, hdr = rw(data_path / file_name, nhdr)     # Reads waveform file
            half_max = min(v) / 2                           # Calculates 50% max
            differential = np.diff(v)                       # Calculates derivative of every point in voltage array
            difference_value = np.abs(v - half_max)   # Finds difference between every point in voltage array & 50% max
            for i in range(0, len(differential)):       # Sets every value in difference_value array with a positive
                if differential[i] > 0:                 # derivative equal to infinity
                    difference_value[i] = np.inf
            index = np.argmin(difference_value)  # Finds index of closest voltage to 50% max with a negative derivative
            half_max_time = t[index]            # Finds time at 50% max
            t2 = t - half_max_time              # Subtracts time of 50% max from time array
            avg = calculate_average(t, v)       # Calculates average baseline
            v2 = v - avg                        # Subtracts average baseline voltage from voltage array
            ww(t2, v2, save_path / file_name, hdr)      # Writes shifted waveform to file
            print('Length of /d1_shifted/:', len(os.listdir(str(save_path))))


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


# Returns 10-90 and 20-80 rise times
def rise_time(t, v, r):
    avg = calculate_average(t, v)       # Calculates average baseline
    t1, t2, charge = calculate_charge(t, v, r)      # Calculates start time of spe
    idx_min_val = np.where(v == min(v))     # Finds index of minimum voltage
    time_min_val = t[idx_min_val]           # Finds time at point of minimum voltage
    min_time = time_min_val[0]

    val10 = .1 * (min(v) - avg)     # Calculates 10% max
    val20 = 2 * val10               # Calculates 20% max
    val80 = 8 * val10               # Calculates 80% max
    val90 = 9 * val10               # Calculates 90% max
    tvals = np.linspace(t1, min_time, 5000)   # Creates array of times from beginning of spe to point of minimum voltage
    vvals = np.interp(tvals, t, v)  # Interpolates & creates array of voltages from beginning of spe to minimum voltage
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


# Returns 10-90 and 20-80 fall times
def fall_time(t, v, r):
    avg = calculate_average(t, v)       # Calculates average baseline
    t1, t2, charge = calculate_charge(t, v, r)      # Calculates end time of spe
    idx_min_val = np.where(v == min(v))     # Finds index of minimum voltage
    time_min_val = t[idx_min_val]       # Finds time at point of minimum voltage
    min_time = time_min_val[0]

    val10 = .1 * (min(v) - avg)     # Calculates 10% max
    val20 = 2 * val10               # Calculates 20% max
    val80 = 8 * val10               # Calculates 80% max
    val90 = 9 * val10               # Calculates 90% max
    tvals = np.linspace(min_time, t2, 5000)     # Creates array of times from point of minimum voltage to end of spe
    vvals = np.interp(tvals, t, v)  # Interpolates & creates array of voltages from point of min voltage to end of spe
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
    fall_time1090 = time10 - time90         # Calculates 10-90 fall time
    fall_time2080 = time20 - time80         # Calculates 20-80 fall time
    fall_time1090 = float(format(fall_time1090, '.2e'))
    fall_time2080 = float(format(fall_time2080, '.2e'))

    return fall_time1090, fall_time2080


# Returns 10%, 20%, 80%, and 90% jitter of spe
def calculate_times(t, v, r):
    avg = calculate_average(t, v)       # Calculates average baseline
    t1, t2, charge = calculate_charge(t, v, r)      # Calculates start time of spe
    idx_min_val = np.where(v == min(v))     # Finds index of minimum voltage
    time_min_val = t[idx_min_val]       # Finds time at point of minimum voltage
    min_time = time_min_val[0]

    val10 = .1 * (min(v) - avg)     # Calculates 10% max
    val20 = 2 * val10               # Calculates 20% max
    val80 = 8 * val10               # Calculates 80% max
    val90 = 9 * val10               # Calculates 90% max
    tvals = np.linspace(t1, min_time, 5000)   # Creates array of times from beginning of spe to point of minimum voltage
    vvals = np.interp(tvals, t, v)   # Interpolates & creates array of voltages from beginning of spe to minimum voltage
    difference_value10 = np.abs(vvals - val10)   # Calculates difference between values in voltage array and 10% max
    difference_value20 = np.abs(vvals - val20)   # Calculates difference between values in voltage array and 20% max
    difference_value80 = np.abs(vvals - val80)   # Calculates difference between values in voltage array and 80% max
    difference_value90 = np.abs(vvals - val90)   # Calculates difference between values in voltage array and 90% max
    index10 = np.argmin(difference_value10)     # Finds index of point of 10% max
    index20 = np.argmin(difference_value20)     # Finds index of point of 20% max
    index80 = np.argmin(difference_value80)     # Finds index of point of 80% max
    index90 = np.argmin(difference_value90)     # Finds index of point of 90% max
    time10 = tvals[index10]         # Finds time of point of 10% max
    time20 = tvals[index20]         # Finds time of point of 20% max
    time80 = tvals[index80]         # Finds time of point of 80% max
    time90 = tvals[index90]         # Finds time of point of 90% max

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
    for item in array:          # Writes an array item on each line of file
        myfile.write(str(item) + '\n')
    myfile.close()


# Calculates beginning & end times of spe waveform, charge, amplitude, fwhm, 10-90 & 20-80 rise times, 10-90 & 20-80
# fall times, and 10%, 20%, 80% & 90% jitter for each spe file
# Returns arrays of beginning & end times of spe waveform, charge, amplitude, fwhm, 10-90 & 20-80 rise times, 10-90 &
# 20-80 fall times, and 10%, 20%, 80% & 90% jitter
def make_arrays(save_shift, dest_path, data_sort, start, end, nhdr, r):
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
            if os.path.isfile(file_name2):      # If the calculations were done previously, they are read from a file
                print("Reading calculations from shifted file #%05d" % i)
                myfile = open(file_name2, 'r')      # Opens file with calculations
                csv_reader = csv.reader(myfile)
                file_array = np.array([])
                for row in csv_reader:      # Creates array with calculation data
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
                # Any spe waveform that returns impossible values is put into the not_spe folder
                if (charge <= 0 or amplitude <= 0 or fwhm <= 0 or rise1090 <= 0 or rise2080 <= 0 or fall1090 <= 0 or
                        fall2080 <= 0 or time10 >= 0 or time20 >= 0 or time80 <= 0 or time90 <= 0):
                    raw_file = str(data_sort / 'C2--waveforms--%05d.txt') % i
                    save_file = str(dest_path / 'not_spe' / 'D1--not_spe--%05d.txt') % i
                    t, v, hdr = rw(raw_file, nhdr)
                    ww(t, v, save_file, hdr)
                    print('Removing file #%05d' % i)
                    os.remove(str(save_shift / 'D1--waveforms--%05d.txt') % i)
                    os.remove(str(dest_path / 'd1_raw' / 'D1--waveforms--%05d.txt') % i)
                    os.remove(str(dest_path / 'calculations' / 'D1--waveforms--%05d.txt') % i)
                # All other spe waveforms' calculations are placed into arrays
                else:
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
            else:           # If the calculations were not done yet, they are calculated
                print("Calculating shifted file #%05d" % i)
                t, v, hdr = rw(file_name1, nhdr)        # Shifted waveform file is read
                t1, t2, charge = calculate_charge(t, v, r)      # Start & end times and charge of spe are calculated
                amplitude = calculate_amp(t, v)     # Amplitude of spe is calculated
                fwhm = calculate_fwhm(t, v)         # FWHM of spe is calculated
                rise1090, rise2080 = rise_time(t, v, r)     # 10-90 & 20-80 rise times of spe are calculated
                fall1090, fall2080 = fall_time(t, v, r)     # 10-90 & 20-80 fall times of spe are calculated
                time10, time20, time80, time90 = calculate_times(t, v, r)   # 10%, 20%, 80% & 90% jitter is calculated
                # Any spe waveform that returns impossible values is put into the not_spe folder
                if (charge <= 0 or amplitude <= 0 or fwhm <= 0 or rise1090 <= 0 or rise2080 <= 0 or fall1090 <= 0 or
                        fall2080 <= 0 or time10 >= 0 or time20 >= 0 or time80 <= 0 or time90 <= 0):
                    raw_file = str(data_sort / 'C2--waveforms--%05d.txt') % i
                    save_file = str(dest_path / 'not_spe' / 'D1--not_spe--%05d.txt') % i
                    t, v, hdr = rw(raw_file, nhdr)
                    ww(t, v, save_file, hdr)
                    print('Removing file #%05d' % i)
                    os.remove(str(save_shift / 'D1--waveforms--%05d.txt') % i)
                    os.remove(str(dest_path / 'd1_raw' / 'D1--waveforms--%05d.txt') % i)
                # All other spe waveforms' calculations are saved in a file & placed into arrays
                else:
                    save_calculations(dest_path, i, t1, t2, charge, amplitude, fwhm, rise1090, rise2080, fall1090,
                                      fall2080, time10, time20, time80, time90)
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
    plt.show()      # Plots histogram with Gaussian fit

    write_hist_data(array, dest_path, filename + '.txt')
