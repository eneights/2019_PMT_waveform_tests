from functions import *


# Creates data set of double spe waveforms (and set of single spe waveforms for comparison)
def create_double_spe(nloops, date, filter_band, nhdr, delay, delay_folder, fsps):
    gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
    save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (date, filter_band))
    dest_path = Path(save_path / 'd2')
    filt_path = Path(dest_path / 'rt_1')
    double_path = Path(dest_path / 'double_spe')
    single_path = Path(dest_path / 'single_spe')

    file_array = np.array([])
    single_file_array = np.array([])
    double_file_array = np.array([])

    # Makes array of all spe file names
    print('Looping through files...')
    for i in range(99999):
        file_name = 'D2--waveforms--%05d.txt' % i
        if os.path.isfile(filt_path / file_name):
            file_array = np.append(file_array, i)

    # Checks for existing double spe files
    print('Checking existing double spe files...')
    for filename in os.listdir(double_path / 'rt_1' / delay_folder):
        print(filename, 'is a file')
        files_added = filename[15:27]
        double_file_array = np.append(double_file_array, files_added)

    # Creates double spe files
    while len(double_file_array) < nloops:
        idx1 = np.random.randint(len(file_array))
        idx2 = np.random.randint(len(file_array))
        file_1 = file_array[idx1]
        file_2 = file_array[idx2]
        files_added = '%05d--%05d' % (file_1, file_2)
        file_name_1 = str(filt_path / 'D2--waveforms--%05d.txt') % file_1
        file_name_2 = str(filt_path / 'D2--waveforms--%05d.txt') % file_2
        t1, v1, hdr1 = rw(file_name_1, nhdr)
        t2, v2, hdr2 = rw(file_name_2, nhdr)
        for j in range(len(t1)):
            t1[j] = float(format(t1[j], '.4e'))
        for j in range(len(t2)):
            t2[j] = float(format(t2[j], '.4e'))
        time_int = float(format(t1[1] - t1[0], '.4e'))
        delay_amt = int(delay / time_int) * time_int
        try:
            if min(t1) > min(t2):
                t2 += delay_amt
            else:
                t1 += delay_amt
            if min(t1) < min(t2):
                idx1 = np.where(t1 == min(t2))[0][0]
                for j in range(idx1):
                    t1 = np.append(t1, float(format(max(t1) + time_int, '.4e')))
                    t2 = np.insert(t2, 0, float(format(min(t2) + time_int, '.4e')))
                    v1 = np.append(v1, 0)
                    v2 = np.insert(v2, 0, 0)
            elif min(t1) > min(t2):
                idx2 = np.where(t2 == min(t1))[0][0]
                for j in range(idx2):
                    t1 = np.insert(t1, 0, float(format(min(t1) - time_int, '.4e')))
                    t2 = np.append(t2, float(format(max(t2) + time_int, '.4e')))
                    v1 = np.insert(v1, 0, 0)
                    v2 = np.append(v2, 0)
            else:
                pass
            t = t1
            v = np.add(v1, v2)
            file_name = 'D2--waveforms--%s.txt' % files_added
            ww(t, v, double_path / 'rt_1' / delay_folder / file_name, hdr1)
            double_file_array = np.append(double_file_array, files_added)
            print('Added files #%05d & #%05d' % (file_1, file_2))
        except Exception:
            pass

    # Checks for existing single spe files
    print('Checking existing single spe files...')
    for filename in os.listdir(single_path / 'rt_1'):
        print(filename, 'is a file')
        file_added = filename[15:20]
        single_file_array = np.append(single_file_array, file_added)

    # Creates single spe files
    while len(single_file_array) < nloops:
        idx = np.random.randint(len(file_array))
        file = file_array[idx]
        print('Adding file #%05d' % file)
        file_added = '%05d' % file
        single_file_array = np.append(single_file_array, file_added)
        file_name_1 = str(filt_path / 'D2--waveforms--%s.txt') % file_added
        t, v, hdr = rw(file_name_1, nhdr)
        file_name_2 = str(single_path / 'rt_1' / 'D2--waveforms--%s.txt') % file_added
        ww(t, v, file_name_2, hdr)

    tau_2 = 1.3279999999999999e-08
    tau_2_2 = 1.035e-08
    tau_2_2_2 = 3.3249999999999997e-08

    factor2 = 2.701641993196675
    factor4 = 3.6693337890689417
    factor8 = 6.6403385193174485

    # For each double spe waveform file, calculates and saves waveforms with 1x, 2x, 4x, and 8x the rise time
    for item in double_file_array:
        file_name = str(double_path / 'rt_1' / delay_folder / 'D2--waveforms--%s.txt') % item
        save_name2 = str(double_path / 'rt_2' / delay_folder / 'D2--waveforms--%s.txt') % item
        save_name4 = str(double_path / 'rt_4' / delay_folder / 'D2--waveforms--%s.txt') % item
        save_name8 = str(double_path / 'rt_8' / delay_folder / 'D2--waveforms--%s.txt') % item

        if os.path.isfile(file_name):
            if os.path.isfile(save_name2) and os.path.isfile(save_name4) and os.path.isfile(save_name8):
                print('File #%s in double_spe_2 folder' % item)
                print('File #%s in double_spe_4 folder' % item)
                print('File #%s in double_spe_8 folder' % item)
            else:
                t, v, hdr = rw(file_name, nhdr)
                v2 = lowpass_filter(v, tau_2, fsps)
                v4 = lowpass_filter(v, tau_2_2, fsps)
                v8 = lowpass_filter(v, tau_2_2_2, fsps)
                v2_gain = v2 * factor2
                v4_gain = v4 * factor4
                v8_gain = v8 * factor8
                ww(t, v2_gain, save_name2, hdr)
                print('File #%s in double_spe_2 folder' % item)
                ww(t, v4_gain, save_name4, hdr)
                print('File #%s in double_spe_4 folder' % item)
                ww(t, v8_gain, save_name8, hdr)
                print('File #%s in double_spe_8 folder' % item)

    # For each single spe waveform file, saves waveforms with 1x, 2x, 4x, and 8x the rise time
    for item in single_file_array:
        file_name = str(single_path / 'rt_1' / 'D2--waveforms--%s.txt') % item
        file_name2 = str(dest_path / 'rt_2' / 'D2--waveforms--%s.txt') % item
        file_name4 = str(dest_path / 'rt_4' / 'D2--waveforms--%s.txt') % item
        file_name8 = str(dest_path / 'rt_8' / 'D2--waveforms--%s.txt') % item
        save_name2 = str(single_path / 'rt_2' / 'D2--waveforms--%s.txt') % item
        save_name4 = str(single_path / 'rt_4' / 'D2--waveforms--%s.txt') % item
        save_name8 = str(single_path / 'rt_8' / 'D2--waveforms--%s.txt') % item

        if os.path.isfile(file_name):
            if os.path.isfile(save_name2):
                print('File #%s in rt_2 folder' % item)
            else:
                t, v, hdr = rw(file_name2, nhdr)
                ww(t, v, save_name2, hdr)
                print('File #%s in rt_2 folder' % item)

        if os.path.isfile(save_name2):
            if os.path.isfile(save_name4):
                print('File #%s in rt_4 folder' % item)
            else:
                t, v, hdr = rw(file_name4, nhdr)
                ww(t, v, save_name4, hdr)
                print('File #%s in rt_4 folder' % item)

        if os.path.isfile(save_name4):
            if os.path.isfile(save_name8):
                print('File #%s in rt_8 folder' % item)
            else:
                t, v, hdr = rw(file_name8, nhdr)
                ww(t, v, save_name8, hdr)
                print('File #%s in rt_8 folder' % item)

    if delay_folder == 'no_delay':
        delay_name = 'no delay'
    elif delay_folder == '0.5x_rt':
        delay_name = '1.52 ns delay'
    elif delay_folder == '1x_rt':
        delay_name = '3.04 ns delay'
    elif delay_folder == '1.5x_rt':
        delay_name = '4.56 ns delay'
    elif delay_folder == '2x_rt':
        delay_name = '6.08 ns delay'
    elif delay_folder == '2.5x_rt':
        delay_name = '7.6 ns delay'
    elif delay_folder == '3x_rt':
        delay_name = '9.12 ns delay'
    elif delay_folder == '3.5x_rt':
        delay_name = '10.6 ns delay'
    elif delay_folder == '4x_rt':
        delay_name = '12.2 ns delay'
    elif delay_folder == '4.5x_rt':
        delay_name = '13.7 ns delay'
    elif delay_folder == '5x_rt':
        delay_name = '15.2 ns delay'
    elif delay_folder == '5.5x_rt':
        delay_name = '16.7 ns delay'
    elif delay_folder == '6x_rt':
        delay_name = '18.2 ns delay'
    elif delay_folder == '40_ns':
        delay_name = '40 ns delay'
    elif delay_folder == '80_ns':
        delay_name = '80 ns delay'
    else:
        delay_name = ''

    print('Calculating averages...')

    idx_tot_1 = 0
    idx_tot_2 = 0
    idx_tot_4 = 0
    idx_tot_8 = 0
    num = 0

    for item in double_file_array:
        num += 1
        file_name = 'D2--waveforms--%s.txt' % item
        t1, v1, hdr1 = rw(double_path / 'rt_1' / delay_folder / file_name, nhdr)    # Reads a waveform file
        t2, v2, hdr2 = rw(double_path / 'rt_2' / delay_folder / file_name, nhdr)    # Reads a waveform file
        t4, v4, hdr4 = rw(double_path / 'rt_4' / delay_folder / file_name, nhdr)    # Reads a waveform file
        t8, v8, hdr8 = rw(double_path / 'rt_8' / delay_folder / file_name, nhdr)    # Reads a waveform file
        idx_tot_1 += int(np.argmin(np.abs(t1)))
        idx_tot_2 += int(np.argmin(np.abs(t2)))
        idx_tot_4 += int(np.argmin(np.abs(t4)))
        idx_tot_8 += int(np.argmin(np.abs(t8)))
    idx_avg_1 = int(idx_tot_1 / num)
    idx_avg_2 = int(idx_tot_1 / num)
    idx_avg_4 = int(idx_tot_1 / num)
    idx_avg_8 = int(idx_tot_1 / num)

    n1 = 0
    n2 = 0
    n4 = 0
    n8 = 0
    len_1 = 0
    len_2 = 0
    len_4 = 0
    len_8 = 0

    for item in double_file_array:
        file_name = 'D2--waveforms--%s.txt' % item
        t1, v1, hdr1 = rw(double_path / 'rt_1' / delay_folder / file_name, nhdr)    # Reads a waveform file
        t2, v2, hdr2 = rw(double_path / 'rt_2' / delay_folder / file_name, nhdr)    # Reads a waveform file
        t4, v4, hdr4 = rw(double_path / 'rt_4' / delay_folder / file_name, nhdr)    # Reads a waveform file
        t8, v8, hdr8 = rw(double_path / 'rt_8' / delay_folder / file_name, nhdr)    # Reads a waveform file
        idx1 = int(np.argmin(np.abs(t1)))
        idx2 = int(np.argmin(np.abs(t2)))
        idx4 = int(np.argmin(np.abs(t4)))
        idx8 = int(np.argmin(np.abs(t8)))
        if idx1 >= idx_avg_1:
            t1 = t1[int(idx_avg_1 / 4):]
            len_1 += len(t1)
            n1 += 1
        if idx2 >= idx_avg_2:
            t2 = t2[int(idx_avg_2 / 4):]
            len_2 += len(t2)
            n2 += 1
        if idx4 >= idx_avg_4:
            t4 = t4[int(idx_avg_4 / 4):]
            len_4 += len(t4)
            n4 += 1
        if idx8 >= idx_avg_8:
            t8 = t8[int(idx_avg_8 / 4):]
            len_8 += len(t8)
            n8 += 1
    len1_avg = int(len_1 / n1)
    len2_avg = int(len_2 / n2)
    len4_avg = int(len_4 / n4)
    len8_avg = int(len_8 / n8)

    # Plots average waveform for double spe
    print('Calculating average double spe waveform...')
    save_file = Path(dest_path / 'plots')
    tsum = 0
    vsum = 0
    n = 0
    for item in double_file_array:
        file_name = 'D2--waveforms--%s.txt' % item
        t, v, hdr = rw(double_path / 'rt_1' / delay_folder / file_name, nhdr)   # Reads a waveform file
        idx = int(np.argmin(np.abs(t)))                                         # Finds index of t = 0 point
        # Only averages waveform files that have enough points before t = 0 & after the spe
        if idx >= idx_avg_1:
            # Removes points before point of minimum t in time & voltage arrays
            t = t[int(idx_avg_1 / 4):]
            v = v[int(idx_avg_1 / 4):]
            if len(t) >= len1_avg:
                # Removes points after chosen point of maximum t in time & voltage arrays
                t = t[:len1_avg]
                v = v[:len1_avg]
                # Sums time & voltage arrays
                tsum += t
                vsum += v
                n += 1
    # Finds average time & voltage arrays
    t_avg = tsum / n
    v_avg = vsum / n
    v_avg = v_avg / min(v_avg)          # Normalizes voltages

    # Plots average waveform & saves image
    plt.plot(t_avg, v_avg)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Voltage')
    plt.title('Average Waveform (' + delay_name + ', no shaping)')
    plt.savefig(save_file / ('avg_waveform_double_rt1_' + delay_folder + '.png'), dpi=360)
    plt.close()

    # Saves average waveform data
    average_file = dest_path / 'hist_data' / ('avg_waveform_double_rt1_' + delay_folder + '.txt')
    hdr = 'Average Waveform\n\n\n\nTime,Ampl\n'
    ww(t_avg, v_avg, average_file, hdr)

    # Plots average waveform for double spe with 2x rise time
    print('Calculating average 2x double spe waveform...')
    save_file = Path(dest_path / 'plots')
    tsum = 0
    vsum = 0
    n = 0
    for item in double_file_array:
        file_name = 'D2--waveforms--%s.txt' % item
        if os.path.isfile(double_path / 'rt_2' / delay_folder / file_name):
            t, v, hdr = rw(double_path / 'rt_2' / delay_folder / file_name, nhdr)  # Reads a waveform file
            idx = int(np.argmin(np.abs(t)))  # Finds index of t = 0 point
            # Only averages waveform files that have enough points before t = 0 & after the spe
            if idx >= idx_avg_2:
                # Removes points before point of minimum t in time & voltage arrays
                t = t[int(idx_avg_2 / 4):]
                v = v[int(idx_avg_2 / 4):]
                if len(t) >= len2_avg:
                    # Removes points after chosen point of maximum t in time & voltage arrays
                    t = t[:len2_avg]
                    v = v[:len2_avg]
                    # Sums time & voltage arrays
                    tsum += t
                    vsum += v
                    n += 1
    # Finds average time & voltage arrays
    t_avg = tsum / n
    v_avg = vsum / n
    v_avg = v_avg / min(v_avg)  # Normalizes voltages

    # Plots average waveform & saves image
    plt.plot(t_avg, v_avg)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Voltage')
    plt.title('Average Waveform (' + delay_name + ', 2x rise time shaping)')
    plt.savefig(save_file / ('avg_waveform_double_rt2_' + delay_folder + '.png'), dpi=360)
    plt.close()

    # Saves average waveform data
    average_file = dest_path / 'hist_data' / ('avg_waveform_double_rt2_' + delay_folder + '.txt')
    hdr = 'Average Waveform\n\n\n\nTime,Ampl\n'
    ww(t_avg, v_avg, average_file, hdr)

    # Plots average waveform for double spe with 4x rise time
    print('Calculating average 4x double spe waveform...')
    save_file = Path(dest_path / 'plots')
    tsum = 0
    vsum = 0
    n = 0
    for item in double_file_array:
        file_name = 'D2--waveforms--%s.txt' % item
        if os.path.isfile(double_path / 'rt_4' / delay_folder / file_name):
            t, v, hdr = rw(double_path / 'rt_4' / delay_folder / file_name, nhdr)  # Reads a waveform file
            idx = int(np.argmin(np.abs(t)))  # Finds index of t = 0 point
            # Only averages waveform files that have enough points before t = 0 & after the spe
            if idx >= idx_avg_4:
                # Removes points before point of minimum t in time & voltage arrays
                t = t[int(idx_avg_4 / 4):]
                v = v[int(idx_avg_4 / 4):]
                if len(t) >= len4_avg:
                    # Removes points after chosen point of maximum t in time & voltage arrays
                    t = t[:len4_avg]
                    v = v[:len4_avg]
                    # Sums time & voltage arrays
                    tsum += t
                    vsum += v
                    n += 1
    # Finds average time & voltage arrays
    t_avg = tsum / n
    v_avg = vsum / n
    v_avg = v_avg / min(v_avg)              # Normalizes voltages

    # Plots average waveform & saves image
    plt.plot(t_avg, v_avg)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Voltage')
    plt.title('Average Waveform (' + delay_name + ', 4x rise time shaping)')
    plt.savefig(save_file / ('avg_waveform_double_rt4_' + delay_folder + '.png'), dpi=360)
    plt.close()

    # Saves average waveform data
    average_file = dest_path / 'hist_data' / ('avg_waveform_double_rt4_' + delay_folder + '.txt')
    hdr = 'Average Waveform\n\n\n\nTime,Ampl\n'
    ww(t_avg, v_avg, average_file, hdr)

    # Plots average waveform for double spe with 8x rise time
    print('Calculating average 8x double spe waveform...')
    save_file = Path(dest_path / 'plots')
    tsum = 0
    vsum = 0
    n = 0
    for item in double_file_array:
        file_name = 'D2--waveforms--%s.txt' % item
        if os.path.isfile(double_path / 'rt_8' / delay_folder / file_name):
            t, v, hdr = rw(double_path / 'rt_8' / delay_folder / file_name, nhdr)  # Reads a waveform file
            idx = int(np.argmin(np.abs(t)))  # Finds index of t = 0 point
            # Only averages waveform files that have enough points before t = 0 & after the spe
            if idx >= idx_avg_8:
                # Removes points before point of minimum t in time & voltage arrays
                t = t[int(idx_avg_8 / 4):]
                v = v[int(idx_avg_8 / 4):]
                if len(t) >= len8_avg:
                    # Removes points after chosen point of maximum t in time & voltage arrays
                    t = t[:len8_avg]
                    v = v[:len8_avg]
                    # Sums time & voltage arrays
                    tsum += t
                    vsum += v
                    n += 1
    # Finds average time & voltage arrays
    t_avg = tsum / n
    v_avg = vsum / n
    v_avg = v_avg / min(v_avg)              # Normalizes voltages

    # Plots average waveform & saves image
    plt.plot(t_avg, v_avg)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Voltage')
    plt.title('Average Waveform (' + delay_name + ', 8x rise time shaping)')
    plt.savefig(save_file / ('avg_waveform_double_rt8_' + delay_folder + '.png'), dpi=360)
    plt.close()

    # Saves average waveform data
    average_file = dest_path / 'hist_data' / ('avg_waveform_double_rt8_' + delay_folder + '.txt')
    hdr = 'Average Waveform\n\n\n\nTime,Ampl\n'
    ww(t_avg, v_avg, average_file, hdr)

    # Calculates 10-90 rise times for each double spe waveform and puts them into arrays
    '''print('Doing calculations...')
    rt_1_array, rt_2_array, rt_4_array, rt_8_array = make_arrays_d(double_file_array, dest_path, delay_folder, dest_path
                                                                   / 'calculations' / 'double_spe', nhdr)

    # Creates histograms of 10-90 rise times for 1x, 2x, 4x, and 8x the initial rise time for double spe waveforms
    print('Creating histograms...')
    plot_histogram(rt_1_array, dest_path, 100, 'Time', '10-90 Rise Time', 's', 'rt_1_double_' + delay_folder)
    plot_histogram(rt_2_array, dest_path, 100, 'Time', '10-90 Rise Time', 's', 'rt_2_double_' + delay_folder)
    plot_histogram(rt_4_array, dest_path, 100, 'Time', '10-90 Rise Time', 's', 'rt_4_double_' + delay_folder)
    plot_histogram(rt_8_array, dest_path, 100, 'Time', '10-90 Rise Time', 's', 'rt_8_double_' + delay_folder)'''


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="create_double_spe", description="Adds spe waveforms")
    parser.add_argument("--nloops", type=int, help='number of double spe files to create (default=1000)', default=1000)
    parser.add_argument("--date", type=int, help='date of data acquisition (default=20190513)', default=20190513)
    parser.add_argument("--fil_band", type=str, help='folder name for data (default=full_bdw_no_nf)',
                        default='full_bdw_no_nf')
    parser.add_argument("--nhdr", type=int, help='number of header lines to skip in raw file (default=5)', default=5)
    parser.add_argument("--delay", type=float, help='delay time (s) (default=0.)', default=0.)
    parser.add_argument("--delay_folder", type=str, help='folder name for delay (default=no_delay)', default='no_delay')
    parser.add_argument("--fsps", type=float, help='samples per second (Hz) (default=20000000000.)',
                        default=20000000000.)
    args = parser.parse_args()

    create_double_spe(args.nloops, args.date, args.fil_band, args.nhdr, args.delay, args.delay_folder, args.fsps)
