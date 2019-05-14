import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# path = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/200MHz_bandwidth/')
path = Path(r'/Users/Eliza/Documents/WATCHMAN/20190514_watchman_spe/')
cwd = os.getcwd()
os.chdir(path)


def read_waveform(file_name, nhdr):
    myfile = open(file_name, 'rb')
    header = []
    x = np.array([])
    y = np.array([])
    for i in range(nhdr):
        header.append(myfile.readline())
    for line in myfile:
        x = np.append(x, float(line.split(str.encode(','))[0]))
        y = np.append(y, float(line.split(str.encode(','))[1]))
    myfile.close()
    return x, y, header


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="read waveform",description="read the waveform datafile.")
    parser.add_argument("--nhdr",type=int,help='number of header lines to skip in the raw file',default=5)
    parser.add_argument("--file_name",type=str,help="filename",default="./C2--waveforms--00000.txt")
    args = parser.parse_args()

    t, v, hdr = read_waveform(args.file_name, args.nhdr)
    print(hdr)
    plt.plot(t, v)
    plt.show()

os.chdir(cwd)