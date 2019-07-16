from functions import *


def amp_cutoff(date, filter_band, nhdr, shaping):
    gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
    save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (date, filter_band))
    dest_path = Path(save_path / 'd2')

    single_file_array = np.array([])
    double_file_array_6x_rt = np.array([])
    cutoff_array = np.array([])
    true_single = np.array([])
    false_single = np.array([])
    true_double = np.array([])
    false_double = np.array([])

    print('Checking existing files...')
    for filename in os.listdir(dest_path / 'single_spe' / shaping):
        print(filename, 'is a file')
        files_added = filename[15:20]
        single_file_array = np.append(single_file_array, files_added)
    for filename in os.listdir(dest_path / 'double_spe' / shaping / '6x_rt'):
        print(filename, 'is a file')
        files_added = filename[15:27]
        double_file_array_6x_rt = np.append(double_file_array_6x_rt, files_added)

    print('Doing calculations...')
    for i in range(-150, 0):
        cutoff_array = np.append(cutoff_array, i)

        x1 = 0
        y1 = 0
        z1 = 0
        x2 = 0
        y2 = 0
        z2 = 0

        for item in single_file_array:
            file_name = str(dest_path / 'single_spe' / shaping / 'D2--waveforms--%s.txt') % item
            t, v, hdr = rw(file_name, nhdr)  # Waveform file is read

            peak_amts = np.array([])

            try:
                z1 += 1
                v_flip = -1 * v
                peaks, _ = signal.find_peaks(v_flip, max(v_flip) / 20)
                for thing in peaks:
                    peak_amts = np.append(peak_amts, v_flip[thing])
                if len(np.where(peak_amts == max(peak_amts))[0]) == 1:
                    peak_amts[np.where(peak_amts == max(peak_amts))] = 0
                else:
                    peak_amts[np.where(peak_amts == max(peak_amts))[0][0]] = 0
                sec_max = v[peaks[np.where(peak_amts == max(peak_amts))]][0]

                if sec_max >= i or len(peaks) == 1:
                    x1 += 1
                else:
                    y1 += 1
            except Exception:
                pass

        for item in double_file_array_6x_rt:
            file_name = str(dest_path / 'double_spe' / shaping / '6x_rt' / 'D2--waveforms--%s.txt') % item
            t, v, hdr = rw(file_name, nhdr)  # Waveform file is read

            peak_amts = np.array([])

            try:
                z2 += 1
                v_flip = -1 * v
                peaks, _ = signal.find_peaks(v_flip, max(v_flip) / 20)
                for thing in peaks:
                    peak_amts = np.append(peak_amts, v_flip[thing])
                if len(np.where(peak_amts == max(peak_amts))[0]) == 1:
                    peak_amts[np.where(peak_amts == max(peak_amts))] = 0
                else:
                    peak_amts[np.where(peak_amts == max(peak_amts))[0][0]] = 0
                sec_max = v[peaks[np.where(peak_amts == max(peak_amts))]][0]

                if sec_max >= i or len(peaks) == 1:
                    x2 += 1
                else:
                    y2 += 1
            except Exception:
                pass

        percent_true_single = x1 / z1 * 100
        percent_false_double = y1 / z1 * 100
        percent_false_single = x2 / z2 * 100
        percent_true_double = y2 / z2 * 100

        true_single = np.append(true_single, percent_true_single)
        false_single = np.append(false_single, percent_false_single)
        true_double = np.append(true_double, percent_true_double)
        false_double = np.append(false_double, percent_false_double)

    idx = np.argmin(np.abs(false_double - 1))
    amp = cutoff_array[idx]
    true_s_per = float(format(true_single[idx], '.2e'))
    false_s_per = float(format(false_single[idx], '.2e'))
    true_d_per = float(format(true_double[idx], '.2e'))
    false_d_per = float(format(false_double[idx], '.2e'))

    print('Making plots...')
    # Plots ROC graphs
    plt.plot(false_single, true_single)
    plt.xlim(-5, 100)
    plt.ylim(-5, 100)
    plt.xlabel('% False Single Peaks')
    plt.ylabel('% True Single Peaks')
    plt.title('ROC Graph (Amplitude Cutoff)\n' + str(false_s_per) + '% false single peaks, ' + str(true_s_per) +
              '% true single peaks')
    plt.plot(false_s_per, true_s_per, marker='x')
    plt.annotate(str(amp) + ' bits', (false_s_per + 3, true_s_per))
    plt.savefig(dest_path / 'plots' / str('roc_single_' + shaping + '.png'),
                dpi=360)
    plt.close()

    plt.plot(false_double, true_double)
    plt.xlim(-5, 100)
    plt.ylim(-5, 100)
    plt.xlabel('% False Double Peaks')
    plt.ylabel('% True Double Peaks')
    plt.title('ROC Graph (Amplitude Cutoff)\n' + str(false_d_per) + '% false double peaks, ' + str(true_d_per) +
              '% true double peaks')
    plt.plot(false_d_per, true_d_per, marker='x')
    plt.annotate(str(amp) + ' bits', (false_d_per + 3, true_d_per))
    plt.savefig(dest_path / 'plots' / str('roc_double_' + shaping + '.png'), dpi=360)
    plt.close()

    # Plots percent error vs amplitude cutoff graphs
    plt.plot(cutoff_array, false_single)
    plt.ylim(-5, 100)
    plt.plot(amp, false_s_per, marker='x')
    plt.xlabel('Amplitude Cutoff (bits)')
    plt.ylabel('% False Single Peaks')
    plt.title('False Single Peaks')
    plt.savefig(dest_path / 'plots' / str('false_single_cutoff_' + shaping + '.png'), dpi=360)
    plt.close()

    plt.plot(cutoff_array, false_double)
    plt.ylim(-5, 100)
    plt.plot(amp, false_d_per, marker='x')
    plt.xlabel('Amplitude Cutoff (bits)')
    plt.ylabel('% False Double Peaks')
    plt.title('False Double Peaks')
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