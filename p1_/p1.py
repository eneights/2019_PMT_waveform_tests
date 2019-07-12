from p1_sort import p1_sort
from functions import *
from info_file import info_file

def p1(start, end, date, date_time, filter_band, nhdr, fsps, fc, numtaps, baseline, r, pmt_hv, gain, offset, trig_delay,
       amp, band, nfilter):
    gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
    save_sort = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (date, filter_band))
    data_sort = Path(save_sort / 'd0')
    dest_path = Path(save_sort / 'd1')
    data_shift = Path(dest_path / 'd1_raw')
    save_shift = Path(dest_path / 'd1_shifted')

    # Separates spes and non-spes into different folders
    print('Sorting files...')
    for i in range(start, end + 1):
        p1_sort(i, nhdr, fsps, fc, numtaps, data_sort, save_sort, baseline)

    # Shifts spes so that when t = 0, v = 50% max and baseline = 0
    print('Shifting waveforms...')
    for i in range(start, end + 1):
        file_name = 'D1--waveforms--%05d.txt' % i
        if os.path.isfile(data_shift / file_name):
            shift_waveform(i, nhdr, data_shift, save_shift)

    # Creates arrays of beginning & end times of spe waveform, time of end of spe, charge, amplitude, fwhm, 10-90 &
    # 20-80 rise times, 10-90 & 20-80 fall times, and 10%, 20%, 80% & 90% jitter
    t1_array, t2_array, charge_array, amplitude_array, fwhm_array, rise1090_array, rise2080_array, fall1090_array, \
        fall2080_array, time10_array, time20_array, time80_array, time90_array = make_arrays(save_shift, dest_path,
                                                                                             data_sort, start, end,
                                                                                             nhdr, r)

    # Plots histograms of charge, amplitude, FWHM, 10-90 & 20-80 rise times, 10-90 & 20-80 fall times, and 10%, 20%, 80%
    # & 90% jitter
    plot_histogram(charge_array, dest_path, 100, 'Charge', 'Charge', 'C', 'charge')
    plot_histogram(amplitude_array, dest_path, 100, 'Voltage', 'Amplitude', 'V', 'amplitude')
    plot_histogram(fwhm_array, dest_path, 100, 'Time', 'FWHM', 's', 'fwhm')
    plot_histogram(rise1090_array, dest_path, 100, 'Time', '10-90 Rise Time', 's', 'rise1090')
    plot_histogram(rise2080_array, dest_path, 100, 'Time', '20-80 Rise Time', 's', 'rise2080')
    plot_histogram(fall1090_array, dest_path, 100, 'Time', '10-90 Fall Time', 's', 'fall1090')
    plot_histogram(fall2080_array, dest_path, 100, 'Time', '20-80 Fall Time', 's', 'fall2080')
    plot_histogram(time10_array, dest_path, 100, 'Time', '10% Jitter', 's', 'time10')
    plot_histogram(time20_array, dest_path, 100, 'Time', '20% Jitter', 's', 'time20')
    plot_histogram(time80_array, dest_path, 100, 'Time', '80% Jitter', 's', 'time80')
    plot_histogram(time90_array, dest_path, 100, 'Time', '90% Jitter', 's', 'time90')

    # Creates d1 info file
    info_file(date_time, data_sort, dest_path, pmt_hv, gain, offset, trig_delay, amp, fsps, band, nfilter, r)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="p1", description="Creating D1")
    parser.add_argument("--start", type=int, help='file number to begin at (default=0)', default=0)
    parser.add_argument("--end", type=int, help='file number to end at (default=99999)', default=99999)
    parser.add_argument("--date", type=int, help='date of data acquisition (YEARMMDD)')
    parser.add_argument("--date_time", type=str, help='date & time of data acquisition')
    parser.add_argument("--fil_band", type=str, help='folder name for data')
    parser.add_argument("--nhdr", type=int, help='number of header lines to skip in raw file (default=5)', default=5)
    parser.add_argument("--fsps", type=float, help='samples per second (Hz) (suggested=20000000000.)')
    parser.add_argument("--fc", type=float, help='filter cutoff frequency (Hz) (default=250000000)', default=250000000.)
    parser.add_argument("--numtaps", type=int, help='filter order + 1 (default=51)', default=51)
    parser.add_argument("--baseline", type=float, help='baseline of data set (V) (suggested=0)')
    parser.add_argument("--r", type=int, help='resistance in ohms (suggested=50)')
    parser.add_argument("--pmt_hv", type=int, help='voltage of PMT (V) (suggested=1800)')
    parser.add_argument("--gain", type=int, help='gain of PMT (suggested=1e7)')
    parser.add_argument("--offset", type=int, help='offset of pulse generator (suggested=0)')
    parser.add_argument("--trig_delay", type=float, help='delay of pulse generator trigger (ns) (suggested=9.)')
    parser.add_argument("--amp", type=float, help='amplitude of pulse generator (V) (suggested=3.5)')
    parser.add_argument("--band", type=str, help='bandwidth of oscilloscope (Hz)')
    parser.add_argument("--nfilter", type=float, help='noise filter on oscilloscope (bits)')
    parser.add_argument("--info_file", type=str, help='path to d0 info file')
    args = parser.parse_args()

    if not args.info_file:
        if not (args.date or args.date_time or args.fil_band or args.fsps or args.baseline or args.r or args.pmt_hv or
                args.gain or args.offset or args.trig_delay or args.amp or args.band or args.nfilter):
            print('Error: Must provide an info file or all other arguments')
        else:
            p1(args.start, args.end, args.date, args.date_time, args.fil_band, args.nhdr, args.fsps, args.fc,
               args.numtaps, args.baseline, args.r, args.pmt_hv, args.gain, args.offset, args.trig_delay, args.amp,
               args.band, args.nfilter)
    else:
        myfile = open(args.info_file, 'r')
        csv_reader = csv.reader(myfile)
        info_array = np.array([])
        path_array = np.array([])
        for row in csv_reader:
            info_array = np.append(info_array, row[1])
        i_date_time = info_array[0]
        i_path = info_array[1]
        i_nhdr = int(info_array[2])
        i_baseline = float(info_array[3])
        i_pmt_hv = int(info_array[4])
        i_gain = int(float(info_array[5]))
        i_offset = int(info_array[6])
        i_trig_delay = float(info_array[7])
        i_amp = float(info_array[8])
        i_fsps = float(info_array[9])
        i_band = info_array[10]
        i_nfilter = float(info_array[11])
        i_r = int(info_array[12])
        a, b, c, d, e, fol, f, i_fil_band, g = i_path.split('/')
        i_date, watch, spe = fol.split('_')
        i_date = int(i_date)

        p1(args.start, args.end, i_date, i_date_time, i_fil_band, i_nhdr, i_fsps, args.fc, args.numtaps,
           i_baseline, i_r, i_pmt_hv, i_gain, i_offset, i_trig_delay, i_amp, i_band, i_nfilter)

        myfile.close()
