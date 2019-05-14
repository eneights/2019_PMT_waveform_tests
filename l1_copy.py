# Fri Dec 21 10:24:07 EST 2018

import numpy as np
import matplotlib.pyplot as plt
import sys
from ROOT import TH1F

###################################################################
# Parameters for the signal
show_plot = False
N = 4002  # Signal window size
fsps = 20000000000.  # Hz, Samples per second
Nloops = 100000
vthr = -0.00025

# Parameters for the filter
fc = 250000000.  # Hz, Filter cutoff frequency
wc = 2. * np.pi * fc / fsps  # Discrete radial frequency
M = 4000  # number of points in the kernel

###################################################################
# 1.) Sinc function for low pass filter
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

# FFT, just to check
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

###################################################################
# Plot the filtered response
nspe = 0
charge = []
time = []
for i in range(Nloops):
    fin = open('/media/tyler/Seagate Expansion Drive/20181212_watchman_spe/C2--waveforms--%05d.txt' % i)
    # print i
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

    # 2.) Subtract the baseline
    y2 = [yi - np.mean(y[0:500]) for yi in y]
    # y3 = [y2i/abs(np.min(y2)) for y2i in y2]
    y3 = y2

    # 3.) Convolve
    y4 = np.convolve(y3, h2)
    # y5 = [y4i/abs(np.min(y4)) for y4i in y4]
    y5 = y4 * .01 / .136
    y6 = y5[2000:6002]

    # 4.) Discriminator
    y7 = y6[500:1000]
    idx = 0.
    for y7i in y7:
        if y7i < vthr:
            nspe += 1
            charge.append(np.mean(len(y7) * (1. / fsps) * y7 / (50. * 1.6 * 10 ** -19), dtype=float))
            time.append(1.E9 * idx / fsps)
            print
            i, nspe
            fout_name = '/media/tyler/Seagate Expansion Drive/20181220_watchman_spe_filter/l1/%05d.txt' % nspe
            fout = open(fout_name, 'w')
            ix = 0.
            for iy6 in y6:
                outstr = '%E,%f\n' % (ix / fsps, iy6)
                fout.write(outstr)
                ix += 1.
            fout.close()
            break
        idx += 1.
    if show_plot:
        plt.plot(y3)
        plt.plot(y6, linewidth=3, color='black')
        plt.ylim(-0.02, 0.01)
        plt.show()

# Histogram charge
plt.hist(charge, 50)
plt.show()

# Histogram time
plt.hist(time, 50)
plt.show()

# Select on basis of charge
t2 = []
ht2 = TH1F('h1', '', 100, 0., 20.)
for ti, qi in zip(time, charge):
    if qi < -2.5E6 and qi > -1.1E7:
        t2.append(ti)
        ht2.Fill(ti)
plt.hist(t2, 50)
plt.show()

ht2.Draw()
raw_input()