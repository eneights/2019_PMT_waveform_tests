import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from shutil import copyfile
from read_waveform import read_waveform as rw
from write_waveform import write_waveform as ww
from p1_sort import p1_sort
from subtract_time import subtract_time
from waveform_filter import waveform_filter

Nloops = 1000
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

for i in range(j, Nloops):
    # spe_name = Path(r'/Users/Eliza/Documents/WATCHMAN/test_files/d1/d1_raw/D1--waveforms--%05d.txt' % i)
    no_spe_name = Path(r'/Users/Eliza/Documents/WATCHMAN/test_files/d1/not_spe/D1--not_spe--%05d.txt' % i)
    # unsure_spe_name = Path(r'/Users/Eliza/Documents/WATCHMAN/test_files/d1/unsure_if_spe/D1--unsure--%05d.txt' % i)
    if os.path.isfile(no_spe_name):
        print(i)
        t, v, hdr = rw(no_spe_name, 5)
        plt.plot(t, v)
        plt.show()
    # elif os.path.isfile(no_spe_name):
        # t, v, hdr = rw(no_spe_name, nhdr)
    # elif os.path.isfile(unsure_spe_name):
        # t, v, hdr = rw(unsure_spe_name, nhdr)
