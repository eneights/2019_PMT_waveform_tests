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
from waveform_filter import waveform_filter

Nloops = 10000
j = 0

for i in range(j, Nloops):
    p1_sort(i)

for i in range(j, Nloops):
    subtract_time(i)

# for i in range(j, Nloops):
    # charge, time = waveform_filter(i)

# plt.hist(charge, 50)        # Charge histogram
# plt.show()

# plt.hist(time, 50)          # Time histogram
# plt.show()


#charge, amplitude, 10-90, 20-80, rise & fall times
