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
    idx_start = random.randint(0, steps - 1)    # Picks a random index to start at
    idx_max = len(v) - 1 - idx_start            # Calculates max value that can be added to starting index
    t_ds = np.array([])
    v_ds = np.array([])
    for i in range(int(idx_max / steps) + 1):  # Creates time & voltage arrays out of values that digitizer would detect
        t_ds = np.append(t_ds, t[idx_start + i])
        v_ds = np.append(v_ds, v[idx_start + i])
    return t_ds, v_ds


# Converts voltage array to bits and adds noise
def digitize(v, noise):
    v_bits = np.array([])
    for i in range(len(v)):
        v_bits = np.append(v_bits, (v[i] * (2 ** 14 - 1) * 2 + 0.5))    # Converts voltage array to bits
    noise_array = np.random.normal(scale=noise, size=len(v_new))        # Creates noise array
    v_digitized = np.add(v_new, noise_array)        # Adds noise to digitized values
    v_digitized = v_digitized.astype(int)
    return v_digitized


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
    plt.close()

    # Saves average waveform data
    file_name = dest_path / 'plots' / (save_name + '.txt')
    hdr = 'Average Waveform\n\n\n\nTime,Ampl\n'
    ww(t_avg, v_avg, file_name, hdr)
