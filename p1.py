import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from shutil import copyfile
from read_waveform import read_waveform as rw
from write_waveform import write_waveform as ww
from p1_sort import p1_sort
from subtract_time import subtract_time

# Signal parameters
show_plot = False
N = 4002                        # Signal window size
fsps = 20000000000.             # Samples per second (Hz)
Nloops = 10000
vthr = -0.00025

# Filter parameters
fc = 250000000.                 # Filter cutoff frequency (Hz)
wc = 2. * np.pi * fc / fsps     # Discrete radial frequency
# M = 4000                      # Number of points in kernel
numtaps = 51                    # Filter order + 1, chosen for balance of good performance and small transient size
lowpass = signal.firwin(numtaps, cutoff=wc/np.pi, window='blackman')    # Blackman windowed lowpass filter

j = 0
for i in range(j, Nloops):
    p1_sort(i)

j = 0
nhdr = 5
for i in range(j, Nloops):
    data_path = Path(r'/Users/Eliza/Documents/WATCHMAN/test_files/d1/d1_raw')
    # data_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/bandwidth/d1/d1_raw')
    save_path = Path(r'/Users/Eliza/Documents/WATCHMAN/test_files/d1/d1_shift')
    # save_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/bandwidth/d1/d1_shift')
    file_name = 'D1--waveforms--%05d.txt' % i
    if os.path.isfile(file_name):
        copyfile(data_path / file_name, save_path / file_name)
        subtract_time(file_name, nhdr)

# Sinc function for low pass filter
n2 = np.arange(-N / 2, N / 2, 1)
h = [np.sin(wc * n2i) / (n2i * np.pi * wc) if n2i != 0 else 1. / np.pi for n2i in n2]
# print 'Sinc function for LPF'
# plt.plot(h)
# plt.show()

# Truncate and zero pad
h2 = h[int(len(h) / 2 - M / 2):int(len(h) / 2 + M / 2)]  # M points around 0
for i in range(4002 - len(h2)):  # pad with zeros
    h2.append(0.)
# print 'Truncated and zero padded'
# plt.plot(h2)
# plt.show()

# Blackman window
blackman_window = [0.54 - 0.46 * np.cos(2 * np.pi * ni / M) for ni in np.arange(len(h2))]
h2 = [h2i * bwi for h2i, bwi in zip(h2, blackman_window)]
# print 'Blackman window = %d' % len(blackman_window)
# plt.plot(h2)
# plt.show()

# FFT
T = N / fsps
df = 1 / T
dw = 2 * np.pi / T
H_FFT = np.fft.fft(h2)
MAG_H_FFT = [abs(IH) for IH in H_FFT[0:(len(h2) / 2 - 1)]]
f = np.fft.fftfreq(N) * N * df / 1.E9
f = f[:len(MAG_H_FFT)]
# print 'FFT of filter kernel'
# print '%d %d' % (len(f),len(MAG_H_FFT))
# plt.plot(f,MAG_H_FFT,'-o')
# plt.xscale('log')
# plt.show()

# Plot filtered response
nspe = 0
charge = []
time = []
for i in range(Nloops):
    path = Path(r'/Users/Eliza/Documents/WATCHMAN/20190514_watchman_spe')
    # path = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/bandwidth/raw')
    fin = open(str(path / 'C2--waveforms--%05d.txt') % i)
    # Header
    for j in range(5):
        fin.readline()
        n = []
        x = []
        y = []
        ni = 0
    for line in fin:
        n.append(ni)
        x.append(float(line.split(',')[0]))
        y.append(float(line.split(',')[1]))
        ni += 1
    fin.close()

    # Subtract baseline
    y2 = [yi - np.mean(y[0:500]) for yi in y]

    # Convolve
    y3 = np.convolve(y2, h2)
    y4 = y3 * .01 / .136
    y5 = y4[2000:6002]

    # Discriminator
    y6 = y5[500:1000]
    idx = 0.
    for y6i in y6:
        if y6i < vthr:
            nspe += 1
            charge.append(np.mean(len(y6) * (1. / fsps) * y6 / (50. * 1.6 * 10 ** -19), dtype=float))
            time.append(1.E9 * idx / fsps)
            print(i, nspe)
            path = Path(r'/Users/Eliza/Documents/WATCHMAN/test_files/d1/d1_shift')
            # path = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/bandwidth/d1')
            fout_name = str(path / '/D1--waveforms--%05d.txt') % nspe
            fout = open(fout_name, 'w')
            ix = 0.
            for iy5 in y5:
                outstr = '%E,%f\n' % (ix / fsps, iy5)
                fout.write(outstr)
                ix += 1.
            fout.close()
            break
        idx += 1.
    if show_plot:
        plt.plot(y2)
        plt.plot(y5, linewidth=3, color='black')
        plt.ylim(-0.02, 0.01)
        plt.show()

# Histogram charge
plt.hist(charge, 50)
plt.show()

# Histogram time
plt.hist(time, 50)
plt.show()

#charge, amplitude, 10-90, 20-80
