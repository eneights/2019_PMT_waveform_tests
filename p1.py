import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from read_waveform import read_waveform as rw
from write_waveform import write_waveform as ww
from p1_sort import p1_sort
from subtract_time import subtract_time
from waveform_filter import waveform_filter
from functions import *

start_file = 650
end_file = 1000
nhdr = 5

t1_array = np.array([])
t2_array = np.array([])
length_array = np.array([])
charge_array = np.array([])
amplitude_array = np.array([])
rise1090_array = np.array([])
rise2080_array = np.array([])
fall1090_array = np.array([])
fall2080_array = np.array([])

for i in range(start_file, end_file + 1):
    p1_sort(i)

# for i in range(start_file, end_file + 1):
    # subtract_time(i)

for i in range(start_file, end_file + 1):
    file_name = Path(r'/Users/Eliza/Documents/WATCHMAN/test_files/d1/d1_shifted/D1--waveforms--%05d.txt' % i)
    if os.path.isfile(file_name):
        print("File: %05d" % i)
        t, v, hdr = rw(file_name, nhdr)
        t1, t2, charge = calculate_charge(t, v)
        if charge < 0:
            print(charge)
        amplitude = calculate_amp(t, v)
        rise1090, rise2080 = rise_time(t, v)
        fall1090, fall2080 = fall_time(t, v)
        t1_array = np.append(t1_array, t1)
        t2_array = np.append(t2_array, t2)
        length_array = np.append(length_array, t2 - t1)
        charge_array = np.append(charge_array, charge)
        amplitude_array = np.append(amplitude_array, amplitude)
        rise1090_array = np.append(rise1090_array, rise1090)
        rise2080_array = np.append(rise2080_array, rise2080)
        fall1090_array = np.append(fall1090_array, fall1090)
        fall2080_array = np.append(fall2080_array, fall2080)

plt.hist(t1_array, 50)
plt.xlabel('Time (s)')
plt.title('Start Time')
plt.show()

plt.hist(t2_array, 50)
plt.xlabel('Time (s)')
plt.title('End Time')
plt.show()

plt.hist(length_array, 50)
plt.xlabel('Time (s)')
plt.title('Timescale of Waveform')
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
