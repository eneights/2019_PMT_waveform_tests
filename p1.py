import os
from pathlib import Path
import numpy as np
from p1_sort import p1_sort
from subtract_time import subtract_time
from functions import *
from info_file import info_file


def p1(start, end, date, filter_band, nhdr, fsps, fc, numtaps, baseline, r, pmt_hv, gain, offset, trig_delay, amp,
       band, nfilter):
    gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
    save_sort = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (date, filter_band))
    data_sort = Path(save_sort / 'd0')
    dest_path = Path(save_sort / 'd1')
    data_shift = Path(dest_path / 'd1_raw')
    save_shift = Path(dest_path / 'd1_shifted')

    for i in range(start, end + 1):
        p1_sort(i, nhdr, fsps, fc, numtaps, data_sort, save_sort, baseline)

    for i in range(start, end + 1):
        file_name = 'D1--waveforms--%05d.txt' % i
        if os.path.isfile(data_shift / file_name):
            subtract_time(i, nhdr, data_shift, save_shift)

    t1_array, t2_array, charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, fall1090_array, \
        fall2080_array, time10_array, time20_array, time80_array, time90_array = make_arrays(save_shift, start, end,
                                                                                             nhdr, r)

    plot_histograms(charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, fall1090_array,
                    fall2080_array, time10_array, time20_array, time80_array, time90_array, dest_path)

    file_name = 'D1--waveforms--%05d.txt' % start
    t, v, hdr = rw(data_sort / file_name, nhdr)
    x = 3 + hdr.find('#1')
    y = hdr.find(',0')
    acq_date_time = hdr[x:y]
    info_file(acq_date_time, data_sort, dest_path, pmt_hv, gain, offset, trig_delay, amp, fsps, band, nfilter, r)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="p1", description="Creating D1")
    parser.add_argument("--start", type=int, help='file number to begin at (default=0)', default=0)
    parser.add_argument("--end", type=int, help='file number to end at (default=99999)', default=99999)
    parser.add_argument("--date", type=int, help='date of data acquisition (YEARMMDD)')
    parser.add_argument("--fil_band", type=str, help='folder name for data')
    parser.add_argument("--nhdr", type=int, help='number of header lines to skip in raw file (default=5)', default=5)
    parser.add_argument("--fsps", type=float, help='samples per second (Hz) (suggested=20000000000.)')
    parser.add_argument("--fc", type=float, help='filter cutoff frequency (Hz) (default=250000000.)', default=250000000.)
    parser.add_argument("--numtaps", type=int, help='filter order + 1 (default=51)', default=51)
    parser.add_argument("--baseline", type=float, help='baseline of data set (V) (suggested=0)')
    parser.add_argument("--r", type=int, help='resistance in ohms (suggested=50)')
    parser.add_argument("--pmt_hv", type=int, help='voltage of PMT (V) (suggested=1800)')
    parser.add_argument("--gain", type=int, help='gain of PMT (suggested=1e7)')
    parser.add_argument("--offset", type=int, help='offset of pulse generator (suggested=0)')
    parser.add_argument("--trig_delay", type=float, help='delay of pulse generator trigger (ns) (suggested=9.)')
    parser.add_argument("--amp", type=float, help='amplitude of pulse generator (V) (suggested=3.5)')
    parser.add_argument("--band", type=str, help='bandwidth of oscilloscope (Hz) (suggested=full)')
    parser.add_argument("--nfilter", type=float, help='noise filter on oscilloscope (bits) (suggested=0)')
    parser.add_argument("--info_file", type=str, help='info file path')
    args = parser.parse_args()

    if not args.info_file:
        if not (args.date or args.fil_band or args.fsps or args.baseline or args.r or args.pmt_hv or args.gain or
                args.offset or args.trig_delay or args.amp or args.band or args.nfilter):
            print('Error: Must provide an info file or all other arguments')
        else:
            p1(args.start, args.end, args.date, args.fil_band, args.nhdr, args.fsps, args.fc, args.numtaps,
               args.baseline, args.r, args.pmt_hv, args.gain, args.offset, args.trig_delay, args.amp, args.band,
               args.nfilter)
    else:
        if not args.fil_band:
            print('Error: Must provide a folder name for data')
        else:
            myfile = open(args.info_file, 'r')
            mylist = list(map(str.strip, myfile.readlines()))
            x = np.array([])
            y = np.array([])
            for item in mylist:
                x = np.append(x, item.split(',')[0])
                # y = np.append(y, item.split(',')[1])
            print(x)
            # print(y)
            # p1(args.start, args.end, date, fil_band, nhdr, fsps, args.fc, args.numtaps, baseline, r, pmt_hv, gain,
            # offset, trig_delay, amp, band, nfilter)
            myfile.close()
