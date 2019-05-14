import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import signal
sys.path.append('../analysis/')
from read_waveform import read_waveform as rw
from write_waveform_copy import write_waveform
import time
import os
from p1_sort import p1_sort

# Signal parameters
show_plot = False
N = 4002                        # Signal window size
fsps = 20000000000.             # Samples per second (Hz)
Nloops = 10000
vthr = -0.00025
# nhdr = 5

# Filter parameters
fc = 250000000.                 # Filter cutoff frequency (Hz)
wc = 2. * np.pi * fc / fsps     # Discrete radial frequency
# M = 4000                      # Number of points in kernel
print('wc', wc)
numtaps = 51                    # Filter order + 1, chosen for balance of good performance and small transient size
lowpass = signal.firwin(numtaps, cutoff=wc/np.pi, window='blackman')    # Blackman windowed lowpass filter

j = 0
for i in range(j, Nloops):
    p1_sort(i)

