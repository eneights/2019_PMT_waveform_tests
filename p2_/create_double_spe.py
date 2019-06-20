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
    for i in range(nloops - len(double_file_array)):
        idx1 = np.random.randint(len(file_array))
        idx2 = np.random.randint(len(file_array))
        file_1 = file_array[idx1]
        file_2 = file_array[idx2]
        print('Adding files #%05d & #%05d' % (file_1, file_2))
        files_added = '%05d--%05d' % (file_1, file_2)
        double_file_array = np.append(double_file_array, files_added)
        file_name_1 = str(filt_path / 'D2--waveforms--%05d.txt') % file_1
        file_name_2 = str(filt_path / 'D2--waveforms--%05d.txt') % file_2
        t1, v1, hdr1 = rw(file_name_1, nhdr)
        t2, v2, hdr2 = rw(file_name_2, nhdr)
        delay_idx = int(delay / (t1[1] - t1[0]))
        delay_amt = delay_idx * (t1[1] - t1[0])
        if min(t1) > min(t2):
            t2 += delay_amt
        else:
            t1 += delay_amt
        for j in range(len(t1)):
            t1[j] = float(format(t1[j], '.4e'))
        for j in range(len(t2)):
            t2[j] = float(format(t2[j], '.4e'))
        min_time = max(min(t1), min(t2))
        min_time = float(format(min_time, '.4e'))
        idx_min_1 = int(np.where(t1 == min_time)[0])
        idx_min_2 = int(np.where(t2 == min_time)[0])
        max_time = min(max(t1), max(t2))
        idx_max_1 = int(np.where(t1 == max_time)[0])
        idx_max_2 = int(np.where(t2 == max_time)[0])
        t = t1[idx_min_1:idx_max_1 + 1]
        v1 = v1[idx_min_1:idx_max_1 + 1]
        v2 = v2[idx_min_2:idx_max_2 + 1]
        v = np.add(v1, v2)
        file_name = 'D2--waveforms--%s.txt' % files_added
        ww(t, v, double_path / 'rt_1' / delay_folder / file_name, hdr1)

    # Checks for existing single spe files
    print('Checking existing single spe files...')
    for filename in os.listdir(single_path / 'rt_1'):
        print(filename, 'is a file')
        file_added = filename[15:20]
        single_file_array = np.append(single_file_array, file_added)

    # Creates single spe files
    for i in range(nloops - len(single_file_array)):
        idx = np.random.randint(len(file_array))
        file = file_array[idx]
        print('Adding file #%05d' % file)
        file_added = '%05d' % file
        single_file_array = np.append(single_file_array, file_added)
        file_name_1 = str(filt_path / 'D2--waveforms--%s.txt') % file_added
        t, v, hdr = rw(file_name_1, nhdr)
        file_name_2 = str(single_path / 'rt_1' / 'D2--waveforms--%s.txt') % file_added
        ww(t, v, file_name_2, hdr)

    '''print('Calculating taus...')
    x1_array = np.array([])
    j_array = np.array([])

    # Uses average spe waveform to calculate tau to use in lowpass filter for 2x rise time
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
    v2 = lowpass_filter(v, tau_2, fsps)  # Creates new average waveform with 2x the rise time

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
    v2_2 = lowpass_filter(v2, tau_2_2, fsps)  # Creates new average waveform with 4x the rise time

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
    tau_2_2_2 = j_array[np.argmin(np.abs(x1_array - 2 * rt1090_2_2))]'''

    tau_2 = 1.3279999999999999e-08
    tau_2_2 = 1.035e-08
    tau_2_2_2 = 3.3249999999999997e-08

    # For each double spe waveform file, calculates and saves waveforms with 1x, 2x, 4x, and 8x the rise time
    for item in double_file_array:
        file_name = str(double_path / 'rt_1' / delay_folder / 'D2--waveforms--%s.txt') % item
        save_name2 = str(double_path / 'rt_2' / delay_folder / 'D2--waveforms--%s.txt') % item
        save_name4 = str(double_path / 'rt_4' / delay_folder / 'D2--waveforms--%s.txt') % item
        save_name8 = str(double_path / 'rt_8' / delay_folder / 'D2--waveforms--%s.txt') % item

        if os.path.isfile(file_name):
            if os.path.isfile(save_name2):
                print('File #%s in double_spe_2 folder' % item)
            else:
                t, v, hdr = rw(file_name, nhdr)
                v2 = lowpass_filter(v, tau_2, fsps)
                ww(t, v2, save_name2, hdr)
                print('File #%s in double_spe_2 folder' % item)

        if os.path.isfile(save_name2):
            if os.path.isfile(save_name4):
                print('File #%s in double_spe_4 folder' % item)
            else:
                t, v, hdr = rw(save_name2, nhdr)
                v4 = lowpass_filter(v, tau_2_2, fsps)
                ww(t, v4, save_name4, hdr)
                print('File #%s in double_spe_4 folder' % item)

        if os.path.isfile(save_name4):
            if os.path.isfile(save_name8):
                print('File #%s in double_spe_8 folder' % item)
            else:
                t, v, hdr = rw(save_name4, nhdr)
                v8 = lowpass_filter(v, tau_2_2_2, fsps)
                ww(t, v8, save_name8, hdr)
                print('File #%s in double_spe_8 folder' % item)

    # For each single spe waveform file, saves waveforms with 1x, 2x, 4x, and 8x the rise time
    for item in single_file_array:
        file_name = str(single_path / 'rt_1' / 'D2--waveforms--%s.txt') % item
        file_name2 = str(dest_path / 'rt_2')
        file_name4 = str(dest_path / 'rt_4')
        file_name8 = str(dest_path / 'rt_8')
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

    # Plots average waveform for double spe
    print('Calculating average double spe waveform...')
    save_file = Path(dest_path / 'plots')
    tsum = 0
    vsum = 0
    n = 0
    for item in double_file_array:
        file_name = 'D2--waveforms--%s.txt' % item
        t, v, hdr = rw(double_path / 'rt_1' / delay_folder / file_name, nhdr)  # Reads a waveform file
        v = v / min(v)  # Normalizes voltages
        idx = int(np.argmin(np.abs(t)))  # Finds index of t = 0 point
        t = np.roll(t, -idx)  # Rolls time array so that t = 0 point is at index 0
        v = np.roll(v, -idx)  # Rolls voltage array so that 50% max point is at index 0
        idx2 = np.where(t == min(t))  # Finds index of point of minimum t
        idx2 = int(idx2[0])
        idx3 = np.where(t == max(t))  # Finds index of point of maximum t
        idx3 = int(idx3[0])
        # Only averages waveform files that have enough points before t = 0 & after the spe
        if idx2 <= 3445:
            # Removes points between point of maximum t & chosen minimum t in time & voltage arrays
            t = np.concatenate((t[:idx3], t[3445:]))
            v = np.concatenate((v[:idx3], v[3445:]))
            # Rolls time & voltage arrays so that point of chosen minimum t is at index 0
            t = np.roll(t, -idx3)
            v = np.roll(v, -idx3)
            if len(t) >= 3800:
                # Removes points after chosen point of maximum t in time & voltage arrays
                t = t[:3800]
                v = v[:3800]
                # Sums time & voltage arrays
                tsum += t
                vsum += v
                n += 1
    # Finds average time & voltage arrays
    t_avg = tsum / n
    v_avg = vsum / n

    # Plots average waveform & saves image
    plt.plot(t_avg, v_avg)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Voltage')
    plt.title('Average Waveform')
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
            v = v / min(v)  # Normalizes voltages
            idx = int(np.argmin(np.abs(t)))  # Finds index of t = 0 point
            t = np.roll(t, -idx)  # Rolls time array so that t = 0 point is at index 0
            v = np.roll(v, -idx)  # Rolls voltage array so that 50% max point is at index 0
            idx2 = np.where(t == min(t))  # Finds index of point of minimum t
            idx2 = int(idx2[0])
            idx3 = np.where(t == max(t))  # Finds index of point of maximum t
            idx3 = int(idx3[0])
            # Only averages waveform files that have enough points before t = 0 & after the spe
            if idx2 <= 3445:
                # Removes points between point of maximum t & chosen minimum t in time & voltage arrays
                t = np.concatenate((t[:idx3], t[3445:]))
                v = np.concatenate((v[:idx3], v[3445:]))
                # Rolls time & voltage arrays so that point of chosen minimum t is at index 0
                t = np.roll(t, -idx3)
                v = np.roll(v, -idx3)
                if len(t) >= 3800:
                    # Removes points after chosen point of maximum t in time & voltage arrays
                    t = t[:3800]
                    v = v[:3800]
                    # Sums time & voltage arrays
                    tsum += t
                    vsum += v
                    n += 1
    # Finds average time & voltage arrays
    t_avg = tsum / n
    v_avg = vsum / n

    # Plots average waveform & saves image
    plt.plot(t_avg, v_avg)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Voltage')
    plt.title('Average Waveform')
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
            v = v / min(v)  # Normalizes voltages
            idx = int(np.argmin(np.abs(t)))  # Finds index of t = 0 point
            t = np.roll(t, -idx)  # Rolls time array so that t = 0 point is at index 0
            v = np.roll(v, -idx)  # Rolls voltage array so that 50% max point is at index 0
            idx2 = np.where(t == min(t))  # Finds index of point of minimum t
            idx2 = int(idx2[0])
            idx3 = np.where(t == max(t))  # Finds index of point of maximum t
            idx3 = int(idx3[0])
            # Only averages waveform files that have enough points before t = 0 & after the spe
            if idx2 <= 3445:
                # Removes points between point of maximum t & chosen minimum t in time & voltage arrays
                t = np.concatenate((t[:idx3], t[3445:]))
                v = np.concatenate((v[:idx3], v[3445:]))
                # Rolls time & voltage arrays so that point of chosen minimum t is at index 0
                t = np.roll(t, -idx3)
                v = np.roll(v, -idx3)
                if len(t) >= 3800:
                    # Removes points after chosen point of maximum t in time & voltage arrays
                    t = t[:3800]
                    v = v[:3800]
                    # Sums time & voltage arrays
                    tsum += t
                    vsum += v
                    n += 1
    # Finds average time & voltage arrays
    t_avg = tsum / n
    v_avg = vsum / n

    # Plots average waveform & saves image
    plt.plot(t_avg, v_avg)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Voltage')
    plt.title('Average Waveform')
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
            v = v / min(v)  # Normalizes voltages
            idx = int(np.argmin(np.abs(t)))  # Finds index of t = 0 point
            t = np.roll(t, -idx)  # Rolls time array so that t = 0 point is at index 0
            v = np.roll(v, -idx)  # Rolls voltage array so that 50% max point is at index 0
            idx2 = np.where(t == min(t))  # Finds index of point of minimum t
            idx2 = int(idx2[0])
            idx3 = np.where(t == max(t))  # Finds index of point of maximum t
            idx3 = int(idx3[0])
            # Only averages waveform files that have enough points before t = 0 & after the spe
            if idx2 <= 3445:
                # Removes points between point of maximum t & chosen minimum t in time & voltage arrays
                t = np.concatenate((t[:idx3], t[3445:]))
                v = np.concatenate((v[:idx3], v[3445:]))
                # Rolls time & voltage arrays so that point of chosen minimum t is at index 0
                t = np.roll(t, -idx3)
                v = np.roll(v, -idx3)
                if len(t) >= 3800:
                    # Removes points after chosen point of maximum t in time & voltage arrays
                    t = t[:3800]
                    v = v[:3800]
                    # Sums time & voltage arrays
                    tsum += t
                    vsum += v
                    n += 1
    # Finds average time & voltage arrays
    t_avg = tsum / n
    v_avg = vsum / n

    # Plots average waveform & saves image
    plt.plot(t_avg, v_avg)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Voltage')
    plt.title('Average Waveform')
    plt.savefig(save_file / ('avg_waveform_double_rt8_' + delay_folder + '.png'), dpi=360)
    plt.close()

    # Saves average waveform data
    average_file = dest_path / 'hist_data' / ('avg_waveform_double_rt8_' + delay_folder + '.txt')
    hdr = 'Average Waveform\n\n\n\nTime,Ampl\n'
    ww(t_avg, v_avg, average_file, hdr)

    # Calculates 10-90 rise times for each double spe waveform and puts them into arrays
    print('Doing calculations...')
    rt_1_array, rt_2_array, rt_4_array, rt_8_array = make_arrays_d(double_file_array, dest_path, delay_folder, dest_path
                                                                   / 'calculations' / 'double_spe', nhdr)

    # Creates histograms of 10-90 rise times for 1x, 2x, 4x, and 8x the initial rise time for double spe waveforms
    print('Creating histograms...')
    plot_histogram(rt_1_array, dest_path, 100, 'Time', '10-90 Rise Time', 's', 'double_spe_rt_1_' + delay_folder)
    plot_histogram(rt_2_array, dest_path, 100, 'Time', '10-90 Rise Time', 's', 'double_spe_rt_2_' + delay_folder)
    plot_histogram(rt_4_array, dest_path, 100, 'Time', '10-90 Rise Time', 's', 'double_spe_rt_4_' + delay_folder)
    plot_histogram(rt_8_array, dest_path, 100, 'Time', '10-90 Rise Time', 's', 'double_spe_rt_8_' + delay_folder)


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
