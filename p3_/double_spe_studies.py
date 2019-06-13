from functions import *


# Downsamples and digitizes double spe waveforms, then calculates charge, amplitude, and FWHM
def double_spe_studies(date, filter_band, nhdr, delay_folder, fsps, fsps_new, noise, r):
    gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
    save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (date, filter_band))
    data_path = Path(save_path / 'd2')
    dest_path = Path(save_path / 'd3')

    double_file_array = np.array([])

    print('Checking existing files...')
    for filename in os.listdir(data_path / 'double_spe' / delay_folder):
        print(filename, 'is a file')
        files_added = filename[15:27]
        double_file_array = np.append(double_file_array, files_added)

    # Copies double spe waveforms with 1x, 2x, 4x, and 8x initial rise times to d3 folder
    for item in double_file_array:
        file_name1 = str(data_path / 'double_spe' / delay_folder / 'D2--waveforms--%s.txt') % item
        file_name2 = str(data_path / 'double_spe_2' / delay_folder / 'D2--waveforms--%s.txt') % item
        file_name4 = str(data_path / 'double_spe_4' / delay_folder / 'D2--waveforms--%s.txt') % item
        file_name8 = str(data_path / 'double_spe_8' / delay_folder / 'D2--waveforms--%s.txt') % item
        save_name1 = str(dest_path / 'double_spe' / delay_folder / 'raw' / 'D3--waveforms--%s.txt') % item
        save_name2 = str(dest_path / 'double_spe_2' / delay_folder / 'raw' / 'D3--waveforms--%s.txt') % item
        save_name4 = str(dest_path / 'double_spe_4' / delay_folder / 'raw' / 'D3--waveforms--%s.txt') % item
        save_name8 = str(dest_path / 'double_spe_8' / delay_folder / 'raw' / 'D3--waveforms--%s.txt') % item

        if os.path.isfile(file_name1):
            if os.path.isfile(save_name1):
                print('File #%s in double_spe folder' % item)
            else:
                t, v, hdr = rw(file_name1, nhdr)
                ww(t, v, save_name1, hdr)
                print('File #%s in double_spe folder' % item)

        if os.path.isfile(file_name2):
            if os.path.isfile(save_name2):
                print('File #%s in double_spe_2 folder' % item)
            else:
                t, v, hdr = rw(file_name2, nhdr)
                ww(t, v, save_name2, hdr)
                print('File #%s in double_spe_2 folder' % item)

        if os.path.isfile(file_name4):
            if os.path.isfile(save_name4):
                print('File #%s in double_spe_4 folder' % item)
            else:
                t, v, hdr = rw(file_name4, nhdr)
                ww(t, v, save_name4, hdr)
                print('File #%s in double_spe_4 folder' % item)

        if os.path.isfile(file_name8):
            if os.path.isfile(save_name8):
                print('File #%s in double_spe_8 folder' % item)
            else:
                t, v, hdr = rw(file_name8, nhdr)
                ww(t, v, save_name8, hdr)
                print('File #%s in double_spe_8 folder' % item)

    # Downsamples waveforms using given fsps
    for item in double_file_array:
        file_name1 = str(dest_path / 'double_spe' / delay_folder / 'raw' / 'D3--waveforms--%s.txt') % item
        file_name2 = str(dest_path / 'double_spe_2' / delay_folder / 'raw' / 'D3--waveforms--%s.txt') % item
        file_name4 = str(dest_path / 'double_spe_4' / delay_folder / 'raw' / 'D3--waveforms--%s.txt') % item
        file_name8 = str(dest_path / 'double_spe_8' / delay_folder / 'raw' / 'D3--waveforms--%s.txt') % item
        save_name1 = str(dest_path / 'double_spe' / delay_folder / 'downsampled' / 'D3--waveforms--%s.txt') % item
        save_name2 = str(dest_path / 'double_spe_2' / delay_folder / 'downsampled' / 'D3--waveforms--%s.txt') % item
        save_name4 = str(dest_path / 'double_spe_4' / delay_folder / 'downsampled' / 'D3--waveforms--%s.txt') % item
        save_name8 = str(dest_path / 'double_spe_8' / delay_folder / 'downsampled' / 'D3--waveforms--%s.txt') % item

        if os.path.isfile(save_name1) and os.path.isfile(save_name2) and os.path.isfile(save_name4) and \
                os.path.isfile(save_name8):
            print('File #%s downsampled' % item)
        else:
            if os.path.isfile(file_name1) and os.path.isfile(file_name2) and os.path.isfile(file_name4) and \
                    os.path.isfile(file_name8):
                print('Downsampling file #%s' % item)
                if not os.path.isfile(save_name1):
                    t, v, hdr = rw(file_name1, nhdr)
                    t_ds, v_ds = downsample(t, v, fsps, fsps_new)
                    ww(t_ds, v_ds, save_name1, hdr)
                if not os.path.isfile(save_name2):
                    t, v, hdr = rw(file_name2, nhdr)
                    t_ds, v_ds = downsample(t, v, fsps, fsps_new)
                    ww(t_ds, v_ds, save_name2, hdr)
                if not os.path.isfile(save_name4):
                    t, v, hdr = rw(file_name4, nhdr)
                    t_ds, v_ds = downsample(t, v, fsps, fsps_new)
                    ww(t_ds, v_ds, save_name4, hdr)
                if not os.path.isfile(save_name8):
                    t, v, hdr = rw(file_name8, nhdr)
                    t_ds, v_ds = downsample(t, v, fsps, fsps_new)
                    ww(t_ds, v_ds, save_name8, hdr)

    # Digitizes waveforms using given noise
    for item in double_file_array:
        file_name1 = str(dest_path / 'double_spe' / delay_folder / 'downsampled' / 'D3--waveforms--%s.txt') % item
        file_name2 = str(dest_path / 'double_spe_2' / delay_folder / 'downsampled' / 'D3--waveforms--%s.txt') % item
        file_name4 = str(dest_path / 'double_spe_4' / delay_folder / 'downsampled' / 'D3--waveforms--%s.txt') % item
        file_name8 = str(dest_path / 'double_spe_8' / delay_folder / 'downsampled' / 'D3--waveforms--%s.txt') % item
        save_name1 = str(dest_path / 'double_spe' / delay_folder / 'digitized' / 'D3--waveforms--%s.txt') % item
        save_name2 = str(dest_path / 'double_spe_2' / delay_folder / 'digitized' / 'D3--waveforms--%s.txt') % item
        save_name4 = str(dest_path / 'double_spe_4' / delay_folder / 'digitized' / 'D3--waveforms--%s.txt') % item
        save_name8 = str(dest_path / 'double_spe_8' / delay_folder / 'digitized' / 'D3--waveforms--%s.txt') % item

        if os.path.isfile(save_name1) and os.path.isfile(save_name2) and os.path.isfile(save_name4) and \
                os.path.isfile(save_name8):
            print('File #%s digitized' % item)
        else:
            if os.path.isfile(file_name1) and os.path.isfile(file_name2) and os.path.isfile(file_name4) and \
                    os.path.isfile(file_name8):
                print('Digitizing file #%s' % item)
                if not os.path.isfile(save_name1):
                    t, v, hdr = rw(file_name1, nhdr)
                    v_dig = digitize(v, noise)
                    ww(t, v_dig, save_name1, hdr)
                if not os.path.isfile(save_name2):
                    t, v, hdr = rw(file_name2, nhdr)
                    v_dig = digitize(v, noise)
                    ww(t, v_dig, save_name2, hdr)
                if not os.path.isfile(save_name4):
                    t, v, hdr = rw(file_name4, nhdr)
                    v_dig = digitize(v, noise)
                    ww(t, v_dig, save_name4, hdr)
                if not os.path.isfile(save_name8):
                    t, v, hdr = rw(file_name8, nhdr)
                    v_dig = digitize(v, noise)
                    ww(t, v_dig, save_name8, hdr)

    # Plots average waveforms for 1x, 2x, 4x, and 8x initial rise times after downsampling and digitizing
    print('Calculating downsampled double spe average waveform...')
    save_file = Path(dest_path / 'plots')
    tsum = 0
    vsum = 0
    n = 0
    for item in double_file_array:
        file_name = 'D3--waveforms--%s.txt' % item
        t, v, hdr = rw(dest_path / 'double_spe' / delay_folder / 'downsampled' / file_name, nhdr)
        v = v / min(v)              # Normalizes voltages
        idx = int(np.argmin(np.abs(t)))         # Finds index of t = 0 point
        t = np.roll(t, -idx)        # Rolls time array so that t = 0 point is at index 0
        v = np.roll(v, -idx)        # Rolls voltage array so that 50% max point is at index 0
        idx2 = np.where(t == min(t))        # Finds index of point of minimum t
        idx2 = int(idx2[0])
        idx3 = np.where(t == max(t))        # Finds index of point of maximum t
        idx3 = int(idx3[0])
        # Only averages waveform files that have enough points before t = 0 & after the spe
        if idx2 <= 84:
            # Removes points between point of maximum t & chosen minimum t in time & voltage arrays
            t = np.concatenate((t[:idx3], t[84:]))
            v = np.concatenate((v[:idx3], v[84:]))
            # Rolls time & voltage arrays so that point of chosen minimum t is at index 0
            t = np.roll(t, -idx3)
            v = np.roll(v, -idx3)
            if len(t) >= 95:
                # Removes points after chosen point of maximum t in time & voltage arrays
                t = t[:95]
                v = v[:95]
                # Sums time & voltage arrays
                tsum += t
                vsum += v
                n += 1
    # Finds average time & voltage arrays
    t_avg = tsum / n
    v_avg = vsum / n

    # Plots average waveform & saves image
    plt.scatter(t_avg, v_avg)
    plt.plot(t_avg, v_avg)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Voltage')
    plt.title('Average Waveform')
    plt.savefig(save_file / ('average_waveform_double_ds_' + delay_folder + '.png'), dpi=360)
    plt.close()

    # Saves average waveform data
    average_file = dest_path / 'hist_data' / ('average_waveform_double_ds_' + delay_folder + '.txt')
    hdr = 'Average Waveform\n\n\n\nTime,Ampl\n'
    ww(t_avg, v_avg, average_file, hdr)

    print('Calculating digitized double spe average waveform...')
    save_file = Path(dest_path / 'plots')
    tsum = 0
    vsum = 0
    n = 0
    for item in double_file_array:
        file_name = 'D3--waveforms--%s.txt' % item
        t, v, hdr = rw(dest_path / 'double_spe' / delay_folder / 'digitized' / file_name, nhdr)
        v = v / min(v)          # Normalizes voltages
        idx = int(np.argmin(np.abs(t)))         # Finds index of t = 0 point
        t = np.roll(t, -idx)        # Rolls time array so that t = 0 point is at index 0
        v = np.roll(v, -idx)        # Rolls voltage array so that 50% max point is at index 0
        idx2 = np.where(t == min(t))        # Finds index of point of minimum t
        idx2 = int(idx2[0])
        idx3 = np.where(t == max(t))        # Finds index of point of maximum t
        idx3 = int(idx3[0])
        # Only averages waveform files that have enough points before t = 0 & after the spe
        if idx2 <= 84:
            # Removes points between point of maximum t & chosen minimum t in time & voltage arrays
            t = np.concatenate((t[:idx3], t[84:]))
            v = np.concatenate((v[:idx3], v[84:]))
            # Rolls time & voltage arrays so that point of chosen minimum t is at index 0
            t = np.roll(t, -idx3)
            v = np.roll(v, -idx3)
            if len(t) >= 95:
                # Removes points after chosen point of maximum t in time & voltage arrays
                t = t[:95]
                v = v[:95]
                # Sums time & voltage arrays
                tsum += t
                vsum += v
                n += 1
    # Finds average time & voltage arrays
    t_avg = tsum / n
    v_avg = vsum / n

    # Plots average waveform & saves image
    plt.scatter(t_avg, v_avg)
    plt.plot(t_avg, v_avg)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Voltage')
    plt.title('Average Waveform')
    plt.savefig(save_file / ('average_waveform_double_dig_' + delay_folder + '.png'), dpi=360)
    plt.close()

    # Saves average waveform data
    average_file = dest_path / 'hist_data' / ('average_waveform_double_dig_' + delay_folder + '.txt')
    hdr = 'Average Waveform\n\n\n\nTime,Ampl\n'
    ww(t_avg, v_avg, average_file, hdr)

    charge_array, amplitude_array, fwhm_array = make_arrays(double_file_array, 'double_spe', delay_folder, dest_path,
                                                            nhdr, r)
    charge_array_2, amplitude_array_2, fwhm_array_2 = make_arrays(double_file_array, 'double_spe_2', delay_folder,
                                                                  dest_path, nhdr, r)
    charge_array_4, amplitude_array_4, fwhm_array_4 = make_arrays(double_file_array, 'double_spe_4', delay_folder,
                                                                  dest_path, nhdr, r)
    charge_array_8, amplitude_array_8, fwhm_array_8 = make_arrays(double_file_array, 'double_spe_8', delay_folder,
                                                                  dest_path, nhdr, r)

    print('Creating histograms...')
    plot_histogram(charge_array, dest_path, 100, 'Charge', 'Charge', 'C', 'charge_double_spe')
    plot_histogram(charge_array_2, dest_path, 100, 'Charge', 'Charge', 'C', 'charge_double_spe_2')
    plot_histogram(charge_array_4, dest_path, 100, 'Charge', 'Charge', 'C', 'charge_double_spe_4')
    plot_histogram(charge_array_8, dest_path, 100, 'Charge', 'Charge', 'C', 'charge_double_spe_8')
    plot_histogram(amplitude_array, dest_path, 100, 'Voltage', 'Amplitude', 'V', 'amp_double_spe')
    plot_histogram(amplitude_array_2, dest_path, 100, 'Voltage', 'Amplitude', 'V', 'amp_double_spe_2')
    plot_histogram(amplitude_array_4, dest_path, 100, 'Voltage', 'Amplitude', 'V', 'amp_double_spe_4')
    plot_histogram(amplitude_array_8, dest_path, 100, 'Voltage', 'Amplitude', 'V', 'amp_double_spe_8')
    plot_histogram(fwhm_array, dest_path, 100, 'Time', 'FWHM', 's', 'fwhm_double_spe')
    plot_histogram(fwhm_array_2, dest_path, 100, 'Time', 'FWHM', 's', 'fwhm_double_spe_2')
    plot_histogram(fwhm_array_4, dest_path, 100, 'Time', 'FWHM', 's', 'fwhm_double_spe_4')
    plot_histogram(fwhm_array_8, dest_path, 100, 'Time', 'FWHM', 's', 'fwhm_double_spe_8')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="double_spe_studies", description="Creating double spe D3")
    parser.add_argument("--date", type=int, help='date of data acquisition (default=20190513)', default=20190513)
    parser.add_argument("--fil_band", type=str, help='folder name for data (default=full_bdw_no_nf)',
                        default='full_bdw_no_nf')
    parser.add_argument("--nhdr", type=int, help='number of header lines to skip in raw file (default=5)', default=5)
    parser.add_argument("--delay_folder", type=str, help='folder name for delay (default=no_delay)', default='no_delay')
    parser.add_argument("--fsps", type=float, help='samples per second (Hz) (default=20000000000.)',
                        default=20000000000.)
    parser.add_argument("--fsps_new", type=float, help='new samples per second (Hz) (default=500000000.)',
                        default=500000000.)
    parser.add_argument("--noise", type=float, help='noise to add (bits) (default=3.30)', default=3.30)
    parser.add_argument("--r", type=int, help='resistance in ohms (default=50)', default=50)
    args = parser.parse_args()

    double_spe_studies(args.date, args.fil_band, args.nhdr, args.delay_folder, args.fsps, args.fsps_new, args.noise,
                       args.r)
