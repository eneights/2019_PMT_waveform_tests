from functions import *
from info_file import info_file


# Creates data sets of spe waveforms with 2x, 4x, and 8x the initial rise times
def p2(start, end, date, date_time, filter_band, nhdr, fsps, r, pmt_hv, gain, offset, trig_delay, amp, band, nfilter):
    gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
    save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (date, filter_band))
    data_path = Path(save_path / 'd1')
    dest_path = Path(save_path / 'd2')
    filt_path1 = Path(dest_path / 'filter1')
    filt_path2 = Path(dest_path / 'filter2')
    filt_path2_2 = Path(dest_path / 'filter2_2')
    filt_path2_2_2 = Path(dest_path / 'filter2_2_2')

    # tau_2 = 1.3e-8
    # tau_2_2 = 1.052e-8
    # tau_2_2_2 = 3.3459999999999997e-8

    print('Calculating taus...')
    x1_array = np.array([])
    j_array = np.array([])

    # Uses average spe waveform to calculate tau to use in lowpass filter for 2x rise time
    average_file = str(data_path / 'hist_data' / 'avg_waveform.txt')
    t, v, hdr = rw(average_file, nhdr)
    v = -1 * v
    rt1090 = rise_time_1090(t, v)
    for i in range(500, 50000):
        j = i * 1e-11
        v_new = lowpass_filter(v, j, fsps)
        x1 = rise_time_1090(t, v_new)
        x1_array = np.append(x1_array, x1)
        j_array = np.append(j_array, j)
        diff_val = x1 - 8 * rt1090
        if diff_val >= 0:
            break
    tau_2 = j_array[np.argmin(np.abs(x1_array - 2 * rt1090))]
    v = -1 * v
    v2 = lowpass_filter(v, tau_2, fsps)     # Creates new average waveform with 2x the rise time

    # Uses average waveform with 2x the rise time to calculate tau to use in lowpass filter for 4x rise time
    v2 = -1 * v2
    x1_array = np.array([])
    j_array = np.array([])
    rt1090_2 = rise_time_1090(t, v2)
    for i in range(500, 50000):
        j = i * 1e-11
        v_new = lowpass_filter(v2, j, fsps)
        x1 = rise_time_1090(t, v_new)
        x1_array = np.append(x1_array, x1)
        j_array = np.append(j_array, j)
        diff_val = x1 - 2 * rt1090_2
        if diff_val >= 0:
            break
    tau_2_2 = j_array[np.argmin(np.abs(x1_array - 2 * rt1090_2))]
    v2 = -1 * v2
    v2_2 = lowpass_filter(v2, tau_2_2, fsps)    # Creates new average waveform with 4x the rise time

    # Uses average waveform with 4x the rise time to calculate tau to use in lowpass filter for 8x rise time
    v2_2 = -1 * v2_2
    x1_array = np.array([])
    j_array = np.array([])
    rt1090_2_2 = rise_time_1090(t, v2_2)
    for i in range(500, 50000):
        j = i * 1e-11
        v_new = lowpass_filter(v2_2, j, fsps)
        x1 = rise_time_1090(t, v_new)
        x1_array = np.append(x1_array, x1)
        j_array = np.append(j_array, j)
        diff_val = x1 - 2 * rt1090_2_2
        if diff_val >= 0:
            break
    tau_2_2_2 = j_array[np.argmin(np.abs(x1_array - 2 * rt1090_2_2))]
    v2_2 = -1 * v2_2
    v2_2_2 = lowpass_filter(v2_2, tau_2_2_2, fsps)      # Creates new average waveform with 8x the rise time

    # Plots average spe waveforms with 1x, 2x, 4x, and 8x the rise time
    plt.plot(t, v)
    plt.plot(t, v2)
    plt.plot(t, v2_2)
    plt.plot(t, v2_2_2)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Voltage')
    plt.title('Average Waveforms\norange tau = ' + str(format(tau_2, '.2e')) + ' s, green tau = ' +
              str(format(tau_2_2, '.2e')) + ' s, red tau = ' + str(format(tau_2_2_2, '.2e')) + ' s')
    plt.savefig(dest_path / 'plots' / 'avg_waveforms_2.png', dpi=360)
    plt.close()

    # For each spe waveform file, calculates and saves waveforms with 1x, 2x, 4x, and 8x the rise time
    for i in range(start, end + 1):
        file_name = str(data_path / 'd1_shifted' / 'D1--waveforms--%05d.txt') % i
        save_name1 = str(filt_path1 / 'D2--waveforms--%05d.txt') % i
        save_name2 = str(filt_path2 / 'D2--waveforms--%05d.txt') % i
        save_name4 = str(filt_path2_2 / 'D2--waveforms--%05d.txt') % i
        save_name8 = str(filt_path2_2_2 / 'D2--waveforms--%05d.txt') % i

        if os.path.isfile(file_name):
            if os.path.isfile(save_name1):
                print('File #%05d in filter1 folder' % i)
            else:
                t, v, hdr = rw(file_name, nhdr)
                ww(t, v, save_name1, hdr)
                print('File #%05d in filter1 folder' % i)

        if os.path.isfile(save_name1):
            if os.path.isfile(save_name2):
                print('File #%05d in filter2 folder' % i)
            else:
                t, v, hdr = rw(save_name1, nhdr)
                v2 = lowpass_filter(v, tau_2, fsps)
                ww(t, v2, save_name2, hdr)
                print('File #%05d in filter2 folder' % i)

        if os.path.isfile(save_name2):
            if os.path.isfile(save_name4):
                print('File #%05d in filter2_2 folder' % i)
            else:
                t, v, hdr = rw(save_name2, nhdr)
                v4 = lowpass_filter(v, tau_2_2, fsps)
                ww(t, v4, save_name4, hdr)
                print('File #%05d in filter2_2 folder' % i)

        if os.path.isfile(save_name4):
            if os.path.isfile(save_name8):
                print('File #%05d in filter2_2_2 folder' % i)
            else:
                t, v, hdr = rw(save_name4, nhdr)
                v8 = lowpass_filter(v, tau_2_2_2, fsps)
                ww(t, v8, save_name8, hdr)
                print('File #%05d in filter2_2_2 folder' % i)

    # Plots average waveform for 1x rise time
    print('Calculating filter1 average waveform...')
    average_file = str(data_path / 'hist_data' / 'avg_waveform.txt')
    t, v, hdr = rw(average_file, nhdr)
    plt.plot(t, v)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Voltage')
    plt.title('Average Waveform')
    plt.savefig(dest_path / 'plots' / 'avg_waveform1.png', dpi=360)
    plt.close()
    ww(t, v, dest_path / 'hist_data' / 'avg_waveform1.txt', 'Average Waveform\n\n\n\nTime,Ampl\n')

    # Plots average waveform for 2x rise time
    print('Calculating filter2 average waveform...')
    average_waveform(start, end, filt_path2, dest_path, nhdr, 'avg_waveform2')

    # Plots average waveform for 4x rise time
    print('Calculating filter2_2 average waveform...')
    average_waveform(start, end, filt_path2_2, dest_path, nhdr, 'avg_waveform2_2')

    # Plots average waveform for 8x rise time
    print('Calculating filter2_2_2 average waveform...')
    average_waveform(start, end, filt_path2_2_2, dest_path, nhdr, 'avg_waveform2_2_2')

    # Calculates 10-90 rise times for each waveform and puts them into arrays
    print('Doing calculations...')
    filter_1_array, filter_2_array, filter_2_2_array, filter_2_2_2_array = \
        make_arrays(dest_path, dest_path / 'calculations', start, end, nhdr)

    # Creates histograms of 10-90 rise times for 1x, 2x, 4x, and 8x the initial rise time
    print('Creating histograms...')
    plot_histogram(filter_1_array, dest_path, 100, 'Time', '10-90 Rise Time', 's', 'filter_1')
    plot_histogram(filter_2_array, dest_path, 100, 'Time', '10-90 Rise Time', 's', 'filter_2')
    plot_histogram(filter_2_2_array, dest_path, 100, 'Time', '10-90 Rise Time', 's', 'filter_2_2')
    plot_histogram(filter_2_2_2_array, dest_path, 100, 'Time', '10-90 Rise Time', 's', 'filter_2_2_2')

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
