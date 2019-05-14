#!/usr/bin/env python

################################################################
# Fri Jan 25 21:31:15 EST 2019
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
def write_waveform(x,y,fname,hdr):
    fout = open(fname,'w')
    for entry in hdr:
        fout.write(entry)
    for ix,iy in zip(x,y):
        line = '%.7E,%f\n' % (ix,iy)
        fout.write(line)
    fout.close()


###############################################################
# For testing
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="write waveform",description="write a waveform datafile.")
    parser.add_argument("--hdr",type=str,help='header string for the output file',default=5)
    parser.add_argument("--fname",type=str,help="filename",default="./C2--waveforms--00000.txt")
    args = parser.parse_args()
    x = range(1000)
    y = [np.random.normal(0,1.) for i in range(1000)]
    write_waveform(x,y,args.fname,args.hdr)