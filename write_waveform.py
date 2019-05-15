import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pathlib import Path
from read_waveform import read_waveform as rw

# path = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/200MHz_bandwidth/')
path = Path(r'/Users/Eliza/Documents/WATCHMAN/20190514_watchman_spe/')
cwd = os.getcwd()
os.chdir(path)

header_default = "LECROYWaveRunner‘N,20113,Waveform\nSegments,1,SegmentSize,4002\n#1," + str(dt.date.today()) + ",0\nTime,Ampl"

def write_waveform(x, y, file_name, hdr):
    fout = open(file_name, 'w')             # opens file
    for entry in str(hdr):
        fout.write(entry)                   # writes header in file
    for ix, iy in zip(x, y):
        line = '%.7E,%f\n' % (ix, iy)       # formats line
        fout.write(line)                    # writes line in file
    fout.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="write waveform", description="write a waveform datafile.")
    parser.add_argument("--hdr", type=str, help='header string for the output file', default=header_default)
    parser.add_argument("--file_name", type=str, help="filename", default="./C2--waveforms--00000.txt")
    args = parser.parse_args()

    x = range(1000)
    y = [np.random.normal(0, 1.) for i in range(1000)]
    write_waveform(x, y, args.file_name, args.hdr)

os.chdir(cwd)