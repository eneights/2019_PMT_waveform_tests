import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from read_waveform import read_waveform as rw
from write_waveform import write_waveform as ww


def subtract_time(file_name, nhdr):
    t, v, hdr = rw(file_name, nhdr)
    half_max = min(v) / 2
    tvals = np.linspace(4.64e-7, 6.64e-7, int(2e6))
    vvals = np.interp(tvals, t, v)
    differential = np.diff(vvals)
    difference_value = np.abs(vvals - half_max)
    for i in range(0, len(differential)):
        if differential[i] > 0:
            difference_value[i] = np.inf
    index = np.argmin(difference_value)
    half_max_time = tvals[index]
    t2 = t - half_max_time
    ww(t2, v, file_name, hdr)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="subtract time", description="subtract time interval from waveform datafile.")
    parser.add_argument("--hdr", type=int, help='number of header lines to skip in the raw file', default=5)
    parser.add_argument("--hdr", type=str, help='header string for the output file', default=5)
    parser.add_argument("--file_name", type=str, help="filename", default="./D1--waveforms--00000.txt")
    args = parser.parse_args()

os.chdir(cwd)
