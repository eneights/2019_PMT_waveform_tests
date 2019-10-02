from functions import *


def amp_cutoff(date, filter_band, nhdr, shaping):
    gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
    save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (date, filter_band))
    dest_path = Path(save_path / 'd2')

    single_file_array = np.array([])
    double_file_array = np.array([])
    cutoff_array = np.array([])
    a1_array = np.array([])
    x1_array = np.array([])
    y1_array = np.array([])
    z1_array = np.array([])
    a2_array = np.array([])
    x2_array = np.array([])
    y2_array = np.array([])
    z2_array = np.array([])
    true_single = np.array([])
    false_single = np.array([])
    true_double = np.array([])
    false_double = np.array([])

    # DELETE LATER
    for filename in os.listdir(dest_path / 'double_spe' / shaping / 'no_delay'):
        print(filename, 'is a file')
        files_added = filename[15:27]
        double_file_array = np.append(double_file_array, files_added)
    for item in double_file_array:
        file_name = str(dest_path / 'double_spe' / shaping / 'no_delay' / 'D2--waveforms--%s.txt') % item
        t, v, hdr = rw(file_name, nhdr)                                         # Waveform file is read
        plt.plot(t, v)
        plt.title(item)
        plt.show()

    print('Checking existing files...')
    for filename in os.listdir(dest_path / 'single_spe' / shaping):
        print(filename, 'is a file')
        files_added = filename[15:20]
        single_file_array = np.append(single_file_array, files_added)
    for filename in os.listdir(dest_path / 'double_spe' / shaping / '80_ns'):
        print(filename, 'is a file')
        files_added = filename[15:27]
        double_file_array = np.append(double_file_array, files_added)

    # single_file_array = single_file_array[:100]
    # double_file_array = double_file_array[:100]

    print('Doing calculations...')
    for i in range(-300, 0):
        voltage = (i - 0.5) / ((2 ** 14 - 1) * 2)
        cutoff_array = np.append(cutoff_array, voltage)

    for i in range(len(cutoff_array)):
        a1_array = np.append(a1_array, 0)
        x1_array = np.append(x1_array, 0)
        y1_array = np.append(y1_array, 0)
        z1_array = np.append(z1_array, 0)
        a2_array = np.append(a2_array, 0)
        x2_array = np.append(x2_array, 0)
        y2_array = np.append(y2_array, 0)
        z2_array = np.append(z2_array, 0)

    for item in single_file_array:
        file_name = str(dest_path / 'single_spe' / shaping / 'D2--waveforms--%s.txt') % item
        t, v, hdr = rw(file_name, nhdr)                                         # Waveform file is read
        v_flip = -1 * v

        for i in cutoff_array:
            idx = np.where(cutoff_array == i)
            z1_array[idx] += 1
            peaks, _ = signal.find_peaks(v_flip, -i)
            if len(peaks) == 0:
                a1_array[idx] += 1
            elif len(peaks) == 1:
                x1_array[idx] += 1
            else:
                diff = 0
                check = 0
                for what in range(1, len(peaks)):
                    if peaks[what] - peaks[what - 1] >= 3:
                        diff += 1
                if diff == 0:
                    x1_array[idx] += 1
                else:
                    idx_peak = np.where(v == max(v))
                    for what in peaks:
                        if what < idx_peak[0][0]:
                            check = 1
                        else:
                            break
                    if check == 0:
                        x1_array[idx] += 1
                    else:
                        y1_array[idx] += 1

    for item in double_file_array:
        file_name = str(dest_path / 'double_spe' / shaping / '80_ns' / 'D2--waveforms--%s.txt') % item
        t, v, hdr = rw(file_name, nhdr)                                         # Waveform file is read
        v_flip = -1 * v

        for i in cutoff_array:
            idx = np.where(cutoff_array == i)
            z2_array[idx] += 1
            peaks, _ = signal.find_peaks(v_flip, -i)
            if len(peaks) == 0:
                a2_array[idx] += 1
            elif len(peaks) == 1:
                x2_array[idx] += 1
            else:
                diff = 0
                check = 0
                for what in range(1, len(peaks)):
                    if peaks[what] - peaks[what - 1] >= 3:
                        diff += 1
                if diff == 0:
                    x2_array[idx] += 1
                else:
                    idx_peak = np.where(v == max(v))
                    for what in peaks:
                        if what < idx_peak[0][0]:
                            check = 1
                        else:
                            break
                    if check == 0:
                        x2_array[idx] += 1
                    else:
                        y2_array[idx] += 1
            if i >= -0.005 and len(peaks) == 1:
                plt.plot(t, v)
                plt.show()

    for item in cutoff_array:
        idx = np.where(cutoff_array == item)

        percent_true_single = x1_array[idx] / z1_array[idx] * 100
        percent_false_double = y1_array[idx] / z1_array[idx] * 100
        percent_false_single = x2_array[idx] / z2_array[idx] * 100
        percent_true_double = y2_array[idx] / z2_array[idx] * 100

        true_single = np.append(true_single, percent_true_single)
        false_single = np.append(false_single, percent_false_single)
        true_double = np.append(true_double, percent_true_double)
        false_double = np.append(false_double, percent_false_double)

    idx = np.argmin(np.abs(false_double - 1))
    amp = float(format(cutoff_array[idx], '.2e'))
    true_s_per = float(format(true_single[idx], '.2e'))
    false_s_per = float(format(false_single[idx], '.2e'))
    true_d_per = float(format(true_double[idx], '.2e'))
    false_d_per = float(format(false_double[idx], '.2e'))

    # Plots percent error vs amplitude cutoff graphs
    print('Making plots...')
    plt.plot(cutoff_array, false_single)
    plt.ylim(-5, 100)
    plt.plot(amp, false_s_per, marker='x')
    plt.xlabel('Amplitude Cutoff (V)')
    plt.ylabel('% False Single Peaks')
    plt.title('False Single Peaks (Amplitude Cutoff = ' + str(amp) + ' V\n' + str(false_s_per) +
              '% false single peaks, ' + str(true_s_per) + '% true single peaks')
    plt.savefig(dest_path / 'plots' / str('false_single_cutoff_' + shaping + '.png'), dpi=360)
    plt.close()

    plt.plot(cutoff_array, false_double)
    plt.ylim(-5, 100)
    plt.plot(amp, false_d_per, marker='x')
    plt.xlabel('Amplitude Cutoff (V)')
    plt.ylabel('% False Double Peaks')
    plt.title('False Double Peaks (Amplitude Cutoff = ' + str(amp) + ' V\n' + str(false_d_per) +
              '% false double peaks, ' + str(true_d_per) + '% true double peaks')
    plt.savefig(dest_path / 'plots' / str('false_double_cutoff_' + shaping + '.png'), dpi=360)
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="amp_cutoff", description="Creates ROC graphs for amplitude cutoff")
    parser.add_argument("--date", type=int, help='date of data acquisition (default=20190513)', default=20190513)
    parser.add_argument("--fil_band", type=str, help='folder name for data (default=full_bdw_no_nf)',
                        default='full_bdw_no_nf')
    parser.add_argument("--nhdr", type=int, help='number of header lines to skip in raw file (default=5)', default=5)
    parser.add_argument("--shaping", type=str, help='shaping amount (default=rt_1)', default='rt_1')
    args = parser.parse_args()

    amp_cutoff(args.date, args.fil_band, args.nhdr, args.shaping)