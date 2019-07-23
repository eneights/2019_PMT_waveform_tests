from functions import *
from info_file import info_file


# Creates data sets of spe waveforms with 2x, 4x, and 8x the initial rise times
def p2(start, end, date, date_time, filter_band, nhdr, fsps, r, pmt_hv, gain, offset, trig_delay, amp, band, nfilter):
    gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
    save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (date, filter_band))
    data_path = Path(save_path / 'd1')
    dest_path = Path(save_path / 'd2')
    filt_path1 = Path(dest_path / 'rt_1')
    filt_path2 = Path(dest_path / 'rt_2')
    filt_path2_2 = Path(dest_path / 'rt_4')
    filt_path2_2_2 = Path(dest_path / 'rt_8')

    print('Calculating taus...')
    # Uses average spe waveform to calculate tau to use in lowpass filter for 2x rise time
    '''average_file = str(data_path / 'hist_data' / 'avg_waveform_d1b.txt')
    t, v, hdr = rw(average_file, nhdr)
    v = -1 * v
    tau_2 = calculate_tau(t, v, fsps)
    v2 = lowpass_filter(v, tau_2, fsps)                     # Creates new average waveform with 2x rise time shaping

    # Uses average waveform with 2x the rise time to calculate tau to use in lowpass filter for 4x rise time
    tau_2_2 = calculate_tau(t, v2, fsps)
    v2_2 = lowpass_filter(v2, tau_2_2, fsps)                # Creates new average waveform with 4x rise time shaping

    # Uses average waveform with 4x the rise time to calculate tau to use in lowpass filter for 8x rise time
    tau_2_2_2 = calculate_tau(t, v2_2, fsps)
    v2_2_2 = lowpass_filter(v2_2, tau_2_2_2, fsps)          # Creates new average waveform with 8x rise time shaping

    tau_2 = 1.3279999999999999e-08
    tau_2_2 = 1.035e-08
    tau_2_2_2 = 3.3249999999999997e-08

    average_file = str(data_path / 'hist_data' / 'avg_waveform_d1b.txt')
    t, v, hdr = rw(average_file, nhdr)
    v = -1 * v
    v2 = lowpass_filter(v, tau_2, fsps)  # Creates new average waveform with 2x rise time shaping
    v2_2 = lowpass_filter(v2, tau_2_2, fsps)  # Creates new average waveform with 4x rise time shaping
    v2_2_2 = lowpass_filter(v2_2, tau_2_2_2, fsps)  # Creates new average waveform with 8x rise time shaping

    # Calculates factors for gain
    amp1 = min(v)
    amp2 = min(v2)
    amp4 = min(v2_2)
    amp8 = min(v2_2_2)
    factor2 = amp1 / amp2
    factor4 = amp1 / amp4
    factor8 = amp1 / amp8

    v_gain = v * -1
    v2_gain = v2 * factor2 * -1
    v2_2_gain = v2_2 * factor4 * -1
    v2_2_2_gain = v2_2_2 * factor8 * -1

    # Plots average spe waveforms with 1x, 2x, 4x, and 8x the rise time
    plt.plot(t, v_gain)
    plt.plot(t, v2_gain)
    plt.plot(t, v2_2_gain)
    plt.plot(t, v2_2_2_gain)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Voltage')
    plt.title('Average Waveforms\norange tau = ' + str(format(tau_2, '.2e')) + ' s, green tau = ' +
              str(format(tau_2_2, '.2e')) + ' s, red tau = ' + str(format(tau_2_2_2, '.2e')) + ' s')
    plt.savefig(dest_path / 'plots' / 'avg_waveforms.png', dpi=360)
    plt.close()'''

    tau_2 = 1.3279999999999999e-08
    tau_2_2 = 1.035e-08
    tau_2_2_2 = 3.3249999999999997e-08

    factor2 = 2.701641993196675
    factor4 = 3.6693337890689417
    factor8 = 6.6403385193174485

    # For each spe waveform file, calculates and saves waveforms with 1x, 2x, 4x, and 8x the rise time
    for i in range(start, end + 1):
        file_name = str(data_path / 'd1b_shifted' / 'D1--waveforms--%05d.txt') % i
        save_name1 = str(filt_path1 / 'D2--waveforms--%05d.txt') % i
        save_name2 = str(filt_path2 / 'D2--waveforms--%05d.txt') % i
        save_name4 = str(filt_path2_2 / 'D2--waveforms--%05d.txt') % i
        save_name8 = str(filt_path2_2_2 / 'D2--waveforms--%05d.txt') % i

        if os.path.isfile(file_name):
            if os.path.isfile(save_name1):
                print('File #%05d in rt_1 folder' % i)
            else:
                t, v, hdr = rw(file_name, nhdr)
                ww(t, v, save_name1, hdr)
                print('File #%05d in rt_1 folder' % i)
            if os.path.isfile(save_name2) and os.path.isfile(save_name4) and os.path.isfile(save_name8):
                print('File #%05d in rt_2 folder' % i)
                print('File #%05d in rt_4 folder' % i)
                print('File #%05d in rt_8 folder' % i)
            else:
                t, v, hdr = rw(save_name1, nhdr)
                v2 = lowpass_filter(v, tau_2, fsps)
                v4 = lowpass_filter(v2, tau_2_2, fsps)
                v8 = lowpass_filter(v4, tau_2_2_2, fsps)
                v2_gain = v2 * factor2
                v4_gain = v4 * factor4
                v8_gain = v8 * factor8
                ww(t, v2_gain, save_name2, hdr)
                print('File #%05d in rt_2 folder' % i)
                ww(t, v4_gain, save_name4, hdr)
                print('File #%05d in rt_4 folder' % i)
                ww(t, v8_gain, save_name8, hdr)
                print('File #%05d in rt_8 folder' % i)

    # Plots average waveform for 1x rise time
    print('Calculating rt_1 average waveform...')
    average_file = str(data_path / 'hist_data' / 'avg_waveform_d1b.txt')
    t, v, hdr = rw(average_file, nhdr)
    plt.plot(t, v)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Voltage')
    plt.title('Average Waveform\nNo shaping')
    plt.savefig(dest_path / 'plots' / 'avg_waveform1.png', dpi=360)
    plt.close()
    ww(t, v, dest_path / 'hist_data' / 'avg_waveform1.txt', 'Average Waveform\n\n\n\nTime,Ampl\n')

    # Plots average waveform for 2x rise time
    print('Calculating rt_2 average waveform...')
    average_waveform(start, end, filt_path2, dest_path, nhdr, 'avg_waveform2', '2x rise time shaping')

    # Plots average waveform for 4x rise time
    print('Calculating rt_4 average waveform...')
    average_waveform(start, end, filt_path2_2, dest_path, nhdr, 'avg_waveform4', '4x rise time shaping')

    # Plots average waveform for 8x rise time
    print('Calculating rt_8 average waveform...')
    average_waveform(start, end, filt_path2_2_2, dest_path, nhdr, 'avg_waveform8', '8x rise time shaping')

    # Calculates 10-90 rise times for each waveform and puts them into arrays
    print('Doing calculations...')
    rt_1_array, rt_2_array, rt_4_array, rt_8_array = make_arrays(dest_path, dest_path / 'calculations' / 'single_spe',
                                                                 start, end, nhdr)

    # Creates histograms of 10-90 rise times for 1x, 2x, 4x, and 8x the initial rise time
    print('Creating histograms...')
    plot_histogram(rt_1_array, dest_path, 100, 'Time', '10-90 Rise Time', 's', 'rt_1_single')
    plot_histogram(rt_2_array, dest_path, 100, 'Time', '10-90 Rise Time', 's', 'rt_2_single')
    plot_histogram(rt_4_array, dest_path, 100, 'Time', '10-90 Rise Time', 's', 'rt_4_single')
    plot_histogram(rt_8_array, dest_path, 100, 'Time', '10-90 Rise Time', 's', 'rt_8_single')

    # Writes info file
    info_file(date_time, data_path, dest_path, pmt_hv, gain, offset, trig_delay, amp, fsps, band, nfilter, r)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="p2", description="Creating D2")
    parser.add_argument("--start", type=int, help='file number to begin at (default=0)', default=0)
    parser.add_argument("--end", type=int, help='file number to end at (default=99999)', default=99999)
    parser.add_argument("--date", type=int, help='date of data acquisition (YEARMMDD)')
    parser.add_argument("--date_time", type=str, help='date & time of d1 processing')
    parser.add_argument("--fil_band", type=str, help='folder name for data')
    parser.add_argument("--nhdr", type=int, help='number of header lines to skip in raw file (default=5)', default=5)
    parser.add_argument("--fsps", type=float, help='samples per second (Hz) (suggested=20000000000.)')
    parser.add_argument("--r", type=int, help='resistance in ohms (suggested=50)')
    parser.add_argument("--pmt_hv", type=int, help='voltage of PMT (V) (suggested=1800)')
    parser.add_argument("--gain", type=int, help='gain of PMT (suggested=1e7)')
    parser.add_argument("--offset", type=int, help='offset of pulse generator (suggested=0)')
    parser.add_argument("--trig_delay", type=float, help='delay of pulse generator trigger (ns) (suggested=9.)')
    parser.add_argument("--amp", type=float, help='amplitude of pulse generator (V) (suggested=3.5)')
    parser.add_argument("--band", type=str, help='bandwidth of oscilloscope (Hz)')
    parser.add_argument("--nfilter", type=float, help='noise filter on oscilloscope (bits)')
    parser.add_argument("--info_file", type=str, help='path to d1 info file')
    args = parser.parse_args()

    if not args.info_file:
        if not (args.date or args.date_time or args.fil_band or args.fsps or args.r or args.pmt_hv or
                args.gain or args.offset or args.trig_delay or args.amp or args.band or args.nfilter):
            print('Error: Must provide an info file or all other arguments')
        else:
            p2(args.start, args.end, args.date, args.date_time, args.fil_band, args.nhdr, args.fsps, args.r,
               args.pmt_hv, args.gain, args.offset, args.trig_delay, args.amp, args.band, args.nfilter)
    else:
        myfile = open(args.info_file, 'r')
        csv_reader = csv.reader(myfile)
        info_array = np.array([])
        path_array = np.array([])
        for row in csv_reader:
            info_array = np.append(info_array, row[1])
        i_date_time = info_array[1]
        i_path = info_array[3]
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

        p2(args.start, args.end, i_date, i_date_time, i_fil_band, args.nhdr, i_fsps, i_r, i_pmt_hv, i_gain, i_offset,
           i_trig_delay, i_amp, i_band, i_nfilter)

        myfile.close()
