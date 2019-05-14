#!/usr/bin/env python

################################################################
# Fri Jan 25 21:30:49 EST 2019
#
# Read a waveform from a time,voltage CSV format
#
# Revision Log:
#
################################################################

import sys
sys.path.append('../')
sys.path.append('')
import numpy as np
import matplotlib.pyplot as plt
import os

###############################################################
# Plot the data
def read_waveform(fname,nhdr):
    fin = open(fname,'r')
    header = []
    x = np.array([])
    y = np.array([])
    for i in range(nhdr):
        header.append(fin.readline())
    for line in fin:
        x = np.append(x,float(line.split(',')[0]))
        y = np.append(y,float(line.split(',')[1]))
    fin.close()
    return x,y,header

###############################################################
# For testing
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="read waveform",description="read the waveform datafile.")
    parser.add_argument("--nhdr",type=int,help='number of header lines to skip in the raw file',default=5)
    parser.add_argument("--fname",type=str,help="filename",default="./C2--waveforms--00000.txt")
    args = parser.parse_args()

    x,y,header = read_waveform(args.fname,args.nhdr)
    print header
    plt.plot(x,y)
    plt.show()