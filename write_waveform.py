import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# path = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/bandwidth/raw')
path = Path(r'/Users/Eliza/Documents/WATCHMAN/20190514_watchman_spe/d1/d1_raw')
cwd = os.getcwd()
os.chdir(path)


def write_waveform(x, y, file_name, hdr):
    myfile = open(file_name, 'w')           # opens file
    for entry in str(hdr):
        myfile.write(entry)                 # writes header in file
    for ix, iy in zip(x, y):
        line = '%.7E,%f\n' % (ix, iy)       # formats line
        myfile.write(line)                  # writes line in file
    myfile.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="write waveform", description="write a waveform datafile.")
    parser.add_argument("--hdr", type=str, help='header string for the output file', default=5)
    parser.add_argument("--file_name", type=str, help="filename", default="./C2--waveforms--00030.txt")
    args = parser.parse_args()

os.chdir(cwd)
