import os
from pathlib import Path
import numpy as np
from p1_sort import p1_sort
from subtract_time import subtract_time
from functions import *


def p1(start, end, nhdr, fsps, fc, numtaps, data_sort, save_sort, data_shift, save_shift, r):
    half_max = np.inf

    for i in range(start, end + 1):
        p1_sort(i, nhdr, fsps, fc, numtaps, data_sort, save_sort)

    for i in range(start, end + 1):
        file_name = 'D1--waveforms--%05d.txt' % i
        if os.path.isfile(shift_data / file_name):
            half_max = subtract_time(i, nhdr, data_shift, save_shift)

    t1_array, t2_array, charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, fall1090_array, \
        fall2080_array = make_arrays(save_shift, start, end, nhdr, r, half_max)

    plot_histograms(charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, fall1090_array,
                    fall2080_array)


if __name__ == '__main__':
    sort_data = Path(r'/Users/Eliza/Documents/WATCHMAN/20190514_watchman_spe')
    # sort_data = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/bandwidth/raw')
    # sort_save = Path(r'/Users/Eliza/Documents/WATCHMAN/test_files')
    sort_save = Path(r'/Users/Eliza/Documents/WATCHMAN/test_files')
    # sort_save = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/bandwidth')
    shift_data = Path(r'/Users/Eliza/Documents/WATCHMAN/test_files/d1/d1_raw')
    shift_save = Path(r'/Users/Eliza/Documents/WATCHMAN/test_files/d1/d1_shifted')
    # shift_data = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/bandwidth/d1/d1_raw')
    # shift_save = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/bandwidth/d1/d1_shifted')
    import argparse
    parser = argparse.ArgumentParser(prog="p1 sort", description="Sorting through raw data to find good SPEs")
    parser.add_argument("--start", type=int, help='file number to begin at', default=0)
    parser.add_argument("--end", type=int, help='file number to end at', default=10828)
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
