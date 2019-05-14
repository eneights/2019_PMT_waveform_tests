# if __name__ == '__main__':
# Imported packages

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import signal
sys.path.append('../analysis/')
from read_waveform import read_waveform as rw
from write_waveform import write_waveform
import time
import os
from p1_sort import p1_sort

################################
# Signal Parameters
show_plot = False
N = 4002              # Signal window size
fsps = 20000000000.   # Hz, Samples per second
Nloops = 100000
vthr = -0.00025
nhdr = 5

# Parameters for the filter
fc = 250000000.       # Hz, Filter cutoff frequency
wc = 2.*np.pi*fc/fsps # Discrete radial frequency
print('wc',wc)
numtaps = 51              # filter order + 1, chosen for balance of good performance and small transient size
lowpass = signal.firwin(numtaps, cutoff = wc/np.pi, window = 'blackman')    # blackman windowed lowpass filter
# frequency response of the filter

# freq, response = signal.freqz(lowpass)
# amp = 20*(np.log10(np.abs(response)))
# plt.figure()
# plt.plot(freq/np.pi , amp, 'b')
# plt.ylabel('Amplitude Response [dB]')
# plt.xlabel('Normalized Frequency')
# plt.title('Transfer function frequency response')
# plt.grid(True)
# plt.show()

j = 57000
for i in range(j, Nloops):
    # print i
    p1_sort(i)
    # j += 1
