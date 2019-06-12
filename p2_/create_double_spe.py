from functions import *


def create_double_spe(nloops, date, filter_band, nhdr, delay, delay_folder, fsps):
    gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
    save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (date, filter_band))
    dest_path = Path(save_path / 'd2')
    filt_path = Path(dest_path / 'filter1')
    path1 = Path(dest_path / 'double_spe')
    path2 = Path(dest_path / 'double_spe_2')
    path4 = Path(dest_path / 'double_spe_4')
    path8 = Path(dest_path / 'double_spe_8')

    file_array = np.array([])
    double_file_array = np.array([])

    print('Looping through files...')
    for i in range(99999):
        file_name = 'D2--waveforms--%05d.txt' % i
        if os.path.isfile(filt_path / file_name):
            file_array = np.append(file_array, i)

    print('Checking existing files...')
    for filename in os.listdir(path1 / delay_folder):
        print(filename, 'is a file')
        files_added = filename[15:27]
        double_file_array = np.append(double_file_array, files_added)

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
        min_time = max(min(t1), min(t2))
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
        ww(t, v, path1 / delay_folder / file_name, hdr1)

    print('Calculating average double spe waveform...')
    save_file = Path(dest_path / 'plots')
    tsum = 0
    vsum = 0
    n = 0
    for item in double_file_array:
        file_name = 'D2--waveforms--%s.txt' % item
        t, v, hdr = rw(path1 / delay_folder / file_name, nhdr)      # Reads a waveform file
        v = v / min(v)                                              # Normalizes voltages
        idx = np.where(t == 0)                                      # Finds index of t = 0 point
        idx = int(idx[0])
        t = np.roll(t, -idx)        # Rolls time array so that t = 0 point is at index 0
        v = np.roll(v, -idx)        # Rolls voltage array so that 50% max point is at index 0
        idx2 = np.where(t == min(t))        # Finds index of point of minimum t
        idx2 = int(idx2[0])
        idx3 = np.where(t == max(t))        # Finds index of point of maximum t
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
    plt.savefig(save_file / ('average_waveform_' + delay_folder + '.png'), dpi=360)
    plt.close()

    # Saves average waveform data
    average_file = dest_path / 'hist_data' / ('average_waveform_' + delay_folder + '.txt')
    hdr = 'Average Waveform\n\n\n\nTime,Ampl\n'
    ww(t_avg, v_avg, average_file, hdr)

    print('Calculating taus...')
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
    tau_2_2_2 = j_array[np.argmin(np.abs(x1_array - 2 * rt1090_2_2))]

    # For each double spe waveform file, calculates and saves waveforms with 1x, 2x, 4x, and 8x the rise time
    for item in double_file_array:
        file_name = str(path1 / delay_folder / 'D2--waveforms--%s.txt') % item
        save_name2 = str(path2 / delay_folder / 'D2--waveforms--%s.txt') % item
        save_name4 = str(path4 / delay_folder / 'D2--waveforms--%s.txt') % item
        save_name8 = str(path8 / delay_folder / 'D2--waveforms--%s.txt') % item

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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="p2", description="Creating D2")
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



