import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# path = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/200MHz_bandwidth/')
path = Path(r'/Users/Eliza/Documents/WATCHMAN/20190514_watchman_spe/')
cwd = os.getcwd()
os.chdir(path)

def read_waveform(file_name, nhdr):
    myfile = open(file_name, 'rb')          # opens file
    header = []                             # creates empty list for header
    x = np.array([])                        # creates empty array for time
    y = np.array([])                        # creates empty array for voltage
    for i in range(nhdr):
        header.append(myfile.readline())    # reads header and adds each line to header list
        header_string = '\n'.join(a.decode("cp1250") for a in header)   # creates string out of header
    for line in myfile:
        x = np.append(x, float(line.split(str.encode(','))[0]))         # fills array with times from file
        y = np.append(y, float(line.split(str.encode(','))[1]))         # fills array with voltages from file
    half_max_value = min(y) / 2
    xvals = np.linspace(4.64e-7, 6.64e-7, int(2e6))
    yvals = np.interp(xvals, x, y)
    differential = np.diff(yvals)
    difference_value = np.abs(yvals - half_max_value)
    for i in range(0, len(differential)):
        if differential[i] > 0:
            difference_value[i] = np.inf
    index = np.argmin(difference_value)
    time = xvals[index]
    myfile.close()
    return x, y, header_string, half_max_value, time


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="read waveform", description="read the waveform datafile.")
    parser.add_argument("--nhdr", type=int, help='number of header lines to skip in the raw file', default=5)
    parser.add_argument("--file_name", type=str, help="filename", default="./C2--waveforms--00030.txt")
    args = parser.parse_args()

    t, v, hdr, half_max, half_max_time = read_waveform(args.file_name, args.nhdr)
    print("\nHeader:\n\n", hdr)
    plt.plot(t, v)
    plt.show()

os.chdir(cwd)