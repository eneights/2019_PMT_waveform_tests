from functions import *


def amp_cutoff(date, filter_band, fsps_new, nhdr, shaping):
    gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
    save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (date, filter_band))
    dest_path = Path(save_path / 'd3')

    single_file_array = np.array([])
    double_file_array_6x_rt = np.array([])
    cutoff_array = np.array([])
    true_single = np.array([])
    false_single = np.array([])
    true_double = np.array([])
    false_double = np.array([])

    print('Checking existing files...')
    for filename in os.listdir(dest_path / 'single_spe' / shaping / str('digitized_' + str(int(fsps_new / 1e6)) +
                                                                        '_Msps')):
        print(filename, 'is a file')
        files_added = filename[15:20]
        single_file_array = np.append(single_file_array, files_added)
    for filename in os.listdir(dest_path / 'double_spe' / shaping / '6x_rt' /
                               str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps')):
        print(filename, 'is a file')
        files_added = filename[15:27]
        double_file_array_6x_rt = np.append(double_file_array_6x_rt, files_added)

    print('Doing calculations...')
    for i in range(-100, 0):
        cutoff_array = np.append(cutoff_array, i)

        x1 = 0
        y1 = 0
        z1 = 0
        x2 = 0
        y2 = 0
        z2 = 0

        for item in single_file_array:
            file_name = str(dest_path / 'single_spe' / shaping / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps')
                            / 'D3--waveforms--%s.txt') % item
            t, v, hdr = rw(file_name, nhdr)                 # Waveform file is read

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
            file_name = str(dest_path / 'double_spe' / shaping / '6x_rt' /
                            str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') / 'D3--waveforms--%s.txt') % item
            t, v, hdr = rw(file_name, nhdr)                 # Waveform file is read

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

    idx_true_s = np.argmin(np.abs(true_single - 80))
    idx_false_s = np.argmin(np.abs(false_single - 20))
    idx_true_d = np.argmin(np.abs(true_double - 80))
    idx_false_d = np.argmin(np.abs(false_double - 20))
    amp_true_s = float(format(cutoff_array[idx_true_s], '.2e'))
    amp_false_s = float(format(cutoff_array[idx_false_s], '.2e'))
    amp_true_d = float(format(cutoff_array[idx_true_d], '.2e'))
    amp_false_d = float(format(cutoff_array[idx_false_d], '.2e'))

    print('Making plots...')
    # Plots ROC graphs
    plt.plot(false_single, true_single)
    plt.xlim(-5, 100)
    plt.ylim(-5, 100)
    plt.plot(false_single[idx_false_s], true_single[idx_false_s], marker='x')
    plt.plot(false_single[idx_true_s], true_single[idx_true_s], marker='x')
    plt.xlabel('% False Single Peaks')
    plt.ylabel('% True Single Peaks')
    plt.title('ROC Graph (Amplitude Cutoff)')
    plt.annotate(str(amp_false_s) + ' bit cutoff', (false_single[idx_false_s] + 3, true_single[idx_false_s]))
    plt.annotate(str(amp_true_s) + ' bit cutoff', (false_single[idx_true_s] + 3, true_single[idx_true_s]))
    plt.savefig(dest_path / 'plots' / str('roc_single_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping + '.png'),
                dpi=360)
    plt.close()

    plt.plot(false_double, true_double)
    plt.xlim(-5, 100)
    plt.ylim(-5, 100)
    plt.plot(false_double[idx_false_s], true_double[idx_false_s], marker='x')
    plt.plot(false_double[idx_true_s], true_double[idx_true_s], marker='x')
    plt.xlabel('% False Double Peaks')
    plt.ylabel('% True Double Peaks')
    plt.title('ROC Graph (Amplitude Cutoff)')
    plt.annotate(str(amp_false_d) + ' bit cutoff', (false_single[idx_false_d] + 3, true_single[idx_false_d]))
    plt.annotate(str(amp_true_d) + ' bit cutoff', (false_single[idx_true_d] + 3, true_single[idx_true_d]))
    plt.savefig(dest_path / 'plots' / str('roc_double_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping + '.png'),
                dpi=360)
    plt.close()

    # Plots percent error vs amplitude cutoff graphs
    plt.plot(cutoff_array, false_single)
    plt.ylim(-5, 100)
    plt.hlines(20, -100, 0)
    plt.xlabel('Amplitude Cutoff (bits)')
    plt.ylabel('% False Single Peaks')
    plt.title('False Single Peaks\nAmplitude Cutoff Min = ' + str(amp_false_s) + ' bits (20% False Single Peaks)')
    plt.savefig(dest_path / 'plots' / str('false_single_cutoff_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                          '.png'), dpi=360)
    plt.close()

    plt.plot(cutoff_array, false_double)
    plt.ylim(-5, 100)
    plt.hlines(20, -100, 0)
    plt.xlabel('Amplitude Cutoff (bits)')
    plt.ylabel('% False Double Peaks')
    plt.title('False Double Peaks\nAmplitude Cutoff Max = ' + str(amp_false_d) + ' bits (20% False Double Peaks)')
    plt.savefig(dest_path / 'plots' / str('false_double_cutoff_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping +
                                          '.png'), dpi=360)
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="amp_cutoff_roc", description="Creates ROC graphs for amplitude cutoff")
    parser.add_argument("--date", type=int, help='date of data acquisition (default=20190513)', default=20190513)
    parser.add_argument("--fil_band", type=str, help='folder name for data (default=full_bdw_no_nf)',
                        default='full_bdw_no_nf')
    parser.add_argument("--fsps_new", type=float, help='new samples per second (Hz) (default=500000000.)',
                        default=500000000.)
    parser.add_argument("--nhdr", type=int, help='number of header lines to skip in raw file (default=5)', default=5)
    parser.add_argument("--shaping", type=str, help='shaping amount (default=rt_1)', default='rt_1')
    args = parser.parse_args()

    amp_cutoff(args.date, args.fil_band, args.fsps_new, args.nhdr, args.shaping)


