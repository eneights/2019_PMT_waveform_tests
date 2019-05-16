import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def waveform_filter(file_num):
    show_plot = False
    N = 4002                                # Signal window size
    fc = 250000000.                         # Hz, Filter cutoff frequency
    fsps = 20000000000.                     # Samples per second (Hz)
    wc = 2. * np.pi * fc / fsps             # Discrete radial frequency
    M = 4000                                # Number of points in kernel
    nspe = 0
    nhdr = 5
    vthr = -0.00025
    charge = []
    time = []

    # data_path = Path(r'/Users/Eliza/Documents/WATCHMAN/test_files/d1/di_shifted')
    # data_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/bandwidth/d1/d1_shifted')
    data_path = Path(r'/Users/Eliza/Documents/WATCHMAN/test_files/test_save/d1/d1_shifted')
    # save_path = Path(r'/Users/Eliza/Documents/WATCHMAN/test_files/d1/d1_filtered')
    # save_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/bandwidth/d1/d1_filtered')
    save_path = Path(r'/Users/Eliza/Documents/WATCHMAN/test_files/test_save/d1/d1_filtered')
    file_name = 'D1--waveforms--%05d.txt' % file_num

    n2 = np.arange(-N / 2, N / 2, 1)       # Sinc function for low pass filter
    h = [np.sin(wc * n2i) / (n2i * np.pi * wc) if n2i != 0 else 1. / np.pi for n2i in n2]       # Truncate and zero pad
    h2 = h[int(len(h) / 2 - M / 2):int(len(h) / 2 + M / 2)]            # M points around 0
    for i in range(N - len(h2)):
        h2.append(0.)
    blackman_window = [0.54 - 0.46 * np.cos(2 * np.pi * ni / M) for ni in np.arange(len(h2))]  # Blackman window
    h2 = [h2i * bwi for h2i, bwi in zip(h2, blackman_window)]

    if os.path.isfile(data_path / file_name):
        myfile = open(data_path / file_name, 'r')
        for j in range(nhdr):
            myfile.readline()
            n = []
            x = []
            y = []
            ni = 0
        for line in myfile:
            n.append(ni)
            x.append(float(line.split(',')[0]))
            y.append(float(line.split(',')[1]))
            ni += 1
        myfile.close()

        y2 = [yi - np.mean(y[0:500]) for yi in y]       # Subtract baseline
        y3 = np.convolve(y2, h2)                        # Convolve
        y4 = y3 * .01 / .136
        y5 = y4[2000:6002]
        y6 = y5[500:1000]                               # Discriminator
        idx = 0.

        for y6i in y6:
            if y6i < vthr:
                nspe += 1
                charge.append(np.mean(len(y6) * (1. / fsps) * y6 / (50. * 1.6 * 10 ** -19), dtype=float))
                time.append(1.E9 * idx / fsps)
                myfile = open(save_path / file_name, 'w')
                ix = 0.
                for iy5 in y5:
                    outstr = '%E,%f\n' % (ix / fsps, iy5)
                    myfile.write(outstr)
                    ix += 1.
                    break
                myfile.close()
                idx += 1
        if show_plot:
            plt.plot(y2)
            plt.plot(y5, linewidth=3, color='black')
            plt.ylim(-0.02, 0.01)
            plt.show()

    return charge, time


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="filter waveform", description="filter data from waveform datafile.")
    parser.add_argument("--file_num", type=int, help='file number to begin at', default=00000)
    args = parser.parse_args()

    c, t = waveform_filter(args.file_num)
