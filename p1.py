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

    time10_array, time20_array, time80_array, time90_array, half_time_array = initial_time_arrays(data_shift, start,
                                                                                                  end, nhdr, r)

    for i in range(start, end + 1):
        file_name = 'D1--waveforms--%05d.txt' % i
        if os.path.isfile(data_shift / file_name):
            half_max = subtract_time(i, nhdr, data_shift, save_shift)

    t1_array, t2_array, charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, fall1090_array, \
        fall2080_array, time10_2_array, time80_2_array, time90_2_array, half_time_2_array = make_arrays(save_shift,
                                                                                                        start, end,
                                                                                                        nhdr, r,
                                                                                                        half_max)

    plot_histograms(charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, fall1090_array,
                    fall2080_array, time10_array, time20_array, time80_array, time90_array, half_time_array,
                    time10_2_array, time80_2_array, time90_2_array, half_time_2_array)

    file_name = 'D1--waveforms--%05d.txt' % start
    t, v, hdr = rw(data_sort / file_name, nhdr)
    x = 3 + hdr.find('#1')
    y = hdr.find(',0')
    acq_date_time = hdr[x:y]
    info_file(acq_date_time, data_sort, dest_path, pmt_hv, gain, offset, trig_delay, amp, fsps, band, nfilter, r)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="p1", description="Creating D1")
    parser.add_argument("--start", type=int, help='file number to begin at', default=0)
    parser.add_argument("--end", type=int, help='file number to end at', default=99999)
    parser.add_argument("--date", type=int, help='date of data acquisition', default=20190513)
    parser.add_argument("--fil_band", type=str, help='folder name for data', default='full_bdw_no_nf')
    parser.add_argument("--nhdr", type=int, help='number of header lines to skip in the raw file', default=5)
    parser.add_argument("--fsps", type=float, help='samples per second (Hz)', default=20000000000.)
    parser.add_argument("--fc", type=float, help='filter cutoff frequency (Hz)', default=250000000.)
    parser.add_argument("--numtaps", type=int, help='filter order + 1', default=51)
    parser.add_argument("--baseline", type=float, help='baseline of data set (V)', default=0)
    parser.add_argument("--r", type=int, help='resistance in ohms', default=50)
    parser.add_argument("--pmt_hv", type=int, help="voltage of PMT", default=1800)
    parser.add_argument("--gain", type=int, help="gain of PMT", default=1e7)
    parser.add_argument("--offset", type=int, help="offset of pulse generator", default=0)
    parser.add_argument("--trig_delay", type=float, help="delay of pulse generator trigger", default=9.)
    parser.add_argument("--amp", type=float, help="amplitude of pulse generator", default=3.5)
    parser.add_argument("--band", type=int, help="bandwidth of oscilloscope", default=0)
    parser.add_argument("--nfilter", type=float, help="noise filter on oscilloscope", default=0)
    args = parser.parse_args()

    p1(args.start, args.end, args.date, args.fil_band, args.nhdr, args.fsps, args.fc, args.numtaps, args.baseline,
       args.r, args.pmt_hv, args.gain, args.offset, args.trig_delay, args.amp, args.band, args.nfilter)
