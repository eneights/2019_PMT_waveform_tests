import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from read_waveform import read_waveform as rw
from p1_sort import p1_sort
from subtract_time import subtract_time
from functions import *


def p1(start, end, nhdr, fsps, fc, numtaps, data_sort, save_sort, data_shift, save_shift, r):
    t1_array = np.array([])
    t2_array = np.array([])
    charge_array = np.array([])
    amplitude_array = np.array([])
    fwhm_array = np.array([])
    rise1090_array = np.array([])
    rise2080_array = np.array([])
    fall1090_array = np.array([])
    fall2080_array = np.array([])

    for i in range(start, end + 1):
        p1_sort(i, nhdr, fsps, fc, numtaps, data_sort, save_sort)

    for i in range(start, end + 1):
        half_max = subtract_time(i, nhdr, data_shift, save_shift)

    for i in range(start, end + 1):
        file_name = save_shift / 'D1--waveforms--%05d.txt' % i
        if os.path.isfile(file_name):
            print("File: %05d" % i)
            t, v, hdr = rw(file_name, nhdr)
            t1, t2, charge = calculate_charge(t, v, r)
            amplitude = calculate_amp(t, v)
            fwhm = calculate_fwhm(t, v, half_max)
            rise1090, rise2080 = rise_time(t, v)
            fall1090, fall2080 = fall_time(t, v)
            t1_array = np.append(t1_array, t1)
            t2_array = np.append(t2_array, t2)
            charge_array = np.append(charge_array, charge)
            amplitude_array = np.append(amplitude_array, amplitude)
            fwhm_array = np.append(fwhm_array, fwhm)
            rise1090_array = np.append(rise1090_array, rise1090)
            rise2080_array = np.append(rise2080_array, rise2080)
            fall1090_array = np.append(fall1090_array, fall1090)
            fall2080_array = np.append(fall2080_array, fall2080)

    plt.hist(fwhm_array, 50)
    plt.xlabel('Time (s)')
    plt.title('FWHM')
    plt.show()

    plt.hist(charge_array, 50)
    plt.xlabel('Charge (C)')
    plt.title('Charge of SPE')
    plt.show()

    plt.hist(amplitude_array, 50)
    plt.xlabel('Voltage (V)')
    plt.title('Amplitude of SPE')
    plt.show()

    plt.hist(rise1090_array, 50)
    plt.xlabel('Time (s)')
    plt.title('10-90 Risetime')
    plt.show()

    plt.hist(rise2080_array, 50)
    plt.xlabel('Time (s)')
    plt.title('20-80 Risetime')
    plt.show()

    plt.hist(fall1090_array, 50)
    plt.xlabel('Time (s)')
    plt.title('10-90 Falltime')
    plt.show()

    plt.hist(fall2080_array, 50)
    plt.xlabel('Time (s)')
    plt.title('20-80 Falltime')
    plt.show()


if __name__ == '__main__':
    sort_data = Path(r'/Users/Eliza/Documents/WATCHMAN/20190514_watchman_spe')
    # sort_data = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/bandwidth/raw')
    # sort_save = Path(r'/Users/Eliza/Documents/WATCHMAN/test_files')
    sort_save = Path(r'/Users/Eliza/Documents/WATCHMAN/test_files/test_sorting')
    # sort_save = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/bandwidth')
    shift_data = Path(r'/Users/Eliza/Documents/WATCHMAN/test_files/d1/d1_raw')
    shift_save = Path(r'/Users/Eliza/Documents/WATCHMAN/test_files/d1/d1_shifted')
    # shift_data = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/bandwidth/d1/d1_raw')
    # shift_save = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/bandwidth/d1/d1_shifted')
    import argparse
    parser = argparse.ArgumentParser(prog="p1 sort", description="Sorting through raw data to find good SPEs")
    parser.add_argument("--start", type=int, help='file number to begin at', default=00000)
    parser.add_argument("--end", type=int, help='file number to end at', default=99999)
    parser.add_argument("--nhdr", type=int, help='number of header lines to skip in the raw file', default=5)
    parser.add_argument("--fsps", type=float, help='samples per second (Hz)', default=20000000000.)
    parser.add_argument("--fc", type=float, help='filter cutoff frequency (Hz)', default=250000000.)
    parser.add_argument("--numtaps", type=int, help='filter order + 1', default=51)
    parser.add_argument("--data_sort", type=str, help='folder to read raw data from', default=sort_data)
    parser.add_argument("--save_sort", type=str, help='folder to save sorted data to', default=sort_save)
    parser.add_argument("--data_shift", type=str, help='folder to read raw D1 data from', default=shift_data)
    parser.add_argument("--save_shift", type=str, help='folder to save baseline shifted data to', default=shift_save)
    parser.add_argument("--r", type=int, help='resistance in ohms', default=50)
    args = parser.parse_args()

    p1(args.start, args.end, args.nhdr, args.fsps, args.fc, args.numtaps, args.data_sort, args.save_sort,
       args.data_shift, args.save_shift, args.r)
