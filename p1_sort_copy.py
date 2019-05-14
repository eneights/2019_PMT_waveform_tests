# Creating the sorting function for p1
# Requires the import of read_waveform and write_waveform to work properly

import sys
sys.path.append('../analysis/')
import numpy as np
import matplotlib.pyplot as plt
from read_waveform import read_waveform as rw
from write_waveform import write_waveform
from scipy import signal
import os




def p1_sort(fnum):
    nhdr = 5
    fsps = 20000000000.   # Hz, Samples per second
    fc = 250000000.       # Hz, Filter cutoff frequency
    wc = 2.*np.pi*fc/fsps # Discrete radial frequency
    numtaps = 51              # filter order + 1, chosen for balance of good performance and small transient size
    lowpass = signal.firwin(numtaps, cutoff = wc/np.pi, window = 'blackman')    # blackman windowed lowpass filter

    fname = 'C:/watchman/data/20181212_watchman_spe/C2--waveforms--%05d.txt' % fnum
    spe_wname = 'C:/watchman/signal_chain_studies/d1/d1_raw/D1--waveforms--%05d.txt' % fnum
    spe_not_there = 'C:/watchman/signal_chain_studies/d1/not_spe/D1--not_spe--%05d.txt' % fnum
    spe_unsure = 'C:/watchman/signal_chain_studies/d1/unsure_if_spe/D1--unsure--%05d.txt' % fnum
    if os.path.isfile(spe_wname):
        pass
    elif os.path.isfile(spe_not_there):
        pass
    elif os.path.isfile(spe_unsure):
        pass
    else:
        (t,v,hdr) = rw(fname,nhdr)

        y = signal.filtfilt(lowpass,1.0, v)
        y2 = y[numtaps:len(y)-1]
        t2 = t[numtaps:len(y)-1]

        y_flip = -1 * y2
        peaks, _ = signal.find_peaks(y_flip, 0.0115, distance = 350)
        y_peaks = y2[peaks]
        t_peaks = t2[peaks]
        y_check = y_peaks <= -0.0117
        y_check_sum = sum(y_check)
        # print(y_check)
        # print('y_check_sum %f' %y_check_sum)

        # print(f)
        if len(peaks) == 1:
            if min(y2[370:1370]) < -0.0125:
                # plt.figure()
                # plt.plot(t,v,'b')
                # plt.plot(t2,y2,'r',linewidth=2.5)
                # plt.plot(t_peaks, y_peaks,'x',color='yellow')
                # plt.grid(True)
                # print('Displaying file #%05d' % fnum)
                # plt.show()
                write_waveform(t2, y2, spe_wname, hdr)
                print(len(os.listdir('../d1/d1_raw/')))
                # print('1')

        else:
            if y_check_sum >= 2:
                if min(y2[370:1370]) < -0.0115:
                    plt.figure()
                    plt.plot(t,v,'b')
                    plt.plot(t2,y2,'r',linewidth=2.5)
                    plt.plot(t_peaks, y_peaks,'x',color='cyan')
                    # plt.plot(t,-0.012*np.ones(len(v)),'k--')
                    plt.grid(True)
                    print('Displaying file #%05d' % fnum)
                    plt.show(block = False)
                    plt.pause(1.5)
                    plt.close()

                    spe_check = 'pre-loop initialization'
                    while spe_check != 'y' and spe_check != 'n' and spe_check != 'u':
                        spe_check = raw_input('Is there a single visible SPE? "y" or "n"\n')
                    if spe_check == 'y':

                        # Write data file to processed SPE folder
                        write_waveform(t2, y2, spe_wname, hdr)
                    elif spe_check == 'n':

                        write_waveform(t2, y2, spe_not_there, hdr)
                    elif spe_check == 'u':

                        write_waveform(t2, y2, spe_unsure, hdr)
                    print('file #%05d: Done' % fnum)
                    # print('mean is %f' % np.mean(y2))
                    print(len(os.listdir('../d1/d1_raw/')))
                    # print('2')
            else:
                if min(y2[370:1370]) < -0.0115:
                    # plt.figure()
                    # plt.plot(t,v,'b')
                    # plt.plot(t2,y2,'r',linewidth=2.5)
                    # plt.plot(t_peaks, y_peaks,'x',color='yellow')
                    # plt.grid(True)
                    # print('Displaying file #%05d' % fnum)
                    # plt.show()
                    write_waveform(t2, y2, spe_wname, hdr)
                    print(len(os.listdir('../d1/d1_raw/')))
                    # print('3')

    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="p1 sort", description="Sorting through raw data to find good SPEs")
    parser.add_argument("--fnum", type = int, help = 'file number to begin at')
    args = parser.parse_args()

    p1_sort(args.fnum)