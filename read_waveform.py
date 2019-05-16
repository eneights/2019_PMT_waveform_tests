import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# path = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/bandwidth/raw')
# path = Path(r'/Users/Eliza/Documents/WATCHMAN/20190514_watchman_spe')
# cwd = os.getcwd()
# os.chdir(path)


def read_waveform(file_name, nhdr):
    header = []  # creates empty list for header
    x = np.array([])  # creates empty array for time
    y = np.array([])  # creates empty array for voltage

    if os.path.isfile(file_name):
        myfile = open(file_name, 'rb')          # opens file
        for i in range(nhdr):
            header.append(myfile.readline())    # reads header and adds each line to header list
        for line in myfile:
            x = np.append(x, float(line.split(str.encode(','))[0]))         # fills array with times from file
            y = np.append(y, float(line.split(str.encode(','))[1]))         # fills array with voltages from file
        myfile.close()
    return x, y, header


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="read waveform", description="read the waveform datafile.")
    parser.add_argument("--nhdr", type=int, help='number of header lines to skip in the raw file', default=5)
    parser.add_argument("--file_name", type=str, help="filename", default="./C2--waveforms--00000.txt")
    args = parser.parse_args()

    t, v, hdr = read_waveform(args.file_name, args.nhdr)
    print("\nHeader:\n\n", hdr)
    plt.plot(t, v)
    plt.show()

# os.chdir(cwd)
