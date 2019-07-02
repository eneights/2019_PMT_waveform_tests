from functions import *


def amp_cutoff_roc(date, filter_band, fsps_new, nhdr, shaping):
    gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
    save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (date, filter_band))
    dest_path = Path(save_path / 'd3')

    double_file_array_no_delay = np.array([])
    double_file_array_6x_rt = np.array([])
    cutoff_array = np.array([])
    true_single = np.array([])
    false_single = np.array([])
    true_multiple = np.array([])
    false_multiple = np.array([])

    print('Checking existing files...')
    for filename in os.listdir(dest_path / 'double_spe' / shaping / 'no_delay' /
                               str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps')):
        print(filename, 'is a file')
        files_added = filename[15:27]
        double_file_array_no_delay = np.append(double_file_array_no_delay, files_added)
    for filename in os.listdir(dest_path / 'double_spe' / shaping / '6x_rt' /
                               str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps')):
        print(filename, 'is a file')
        files_added = filename[15:27]
        double_file_array_6x_rt = np.append(double_file_array_6x_rt, files_added)

    for i in range(-100, -10):
        cutoff_array = np.append(cutoff_array, i)

        x1 = 0
        y1 = 0
        z1 = 0
        x2 = 0
        y2 = 0
        z2 = 0

        for item in double_file_array_no_delay:
            file_name = str(dest_path / 'double_spe' / shaping / 'no_delay' /
                            str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') / 'D3--waveforms--%s.txt') % item
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
        percent_false_multiple = y1 / z1 * 100
        percent_false_single = x2 / z2 * 100
        percent_true_multiple = y2 / z2 * 100

        true_single = np.append(true_single, percent_true_single)
        false_single = np.append(false_single, percent_false_single)
        true_multiple = np.append(true_multiple, percent_true_multiple)
        false_multiple = np.append(false_multiple, percent_false_multiple)

    # Plots ROC graphs for double waveforms with no delay
    plt.plot(false_single, true_single)
    plt.xlim(-5, 100)
    plt.ylim(-5, 100)
    plt.xlabel('% False Single Peaks')
    plt.ylabel('% True Single Peaks')
    plt.title('ROC Graph (Minimum Amplitude Cutoff)')
    plt.savefig(dest_path / 'plots' / str('roc_single_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping + '.png'),
                dpi=360)
    plt.close()

    plt.plot(false_multiple, true_multiple)
    plt.xlim(-5, 100)
    plt.ylim(-5, 100)
    plt.xlabel('% False Multiple Peaks')
    plt.ylabel('% True Multiple Peaks')
    plt.title('ROC Graph (Minimum Amplitude Cutoff)')
    plt.savefig(dest_path / 'plots' / str('roc_multiple_' + str(int(fsps_new / 1e6)) + '_Msps_' + shaping + '.png'),
                dpi=360)
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

    amp_cutoff_roc(args.date, args.fil_band, args.fsps_new, args.nhdr, args.shaping)


