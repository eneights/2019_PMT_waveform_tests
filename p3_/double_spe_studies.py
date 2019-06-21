from functions import *


# Downsamples and digitizes double spe waveforms, then calculates charge, amplitude, and FWHM
def double_spe_studies(date, filter_band, nhdr, delay_folder, fsps, fsps_new, noise, r):
    gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
    save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (date, filter_band))
    data_path = Path(save_path / 'd2')
    dest_path = Path(save_path / 'd3')

    single_file_array = np.array([])
    double_file_array = np.array([])

    print('Checking existing files...')
    for filename in os.listdir(data_path / 'double_spe' / 'rt_1' / delay_folder):
        print(filename, 'is a file')
        files_added = filename[15:27]
        double_file_array = np.append(double_file_array, files_added)

    for filename in os.listdir(data_path / 'single_spe' / 'rt_1'):
        print(filename, 'is a file')
        file_added = filename[15:20]
        single_file_array = np.append(single_file_array, file_added)

    # Copies double spe waveforms with 1x, 2x, 4x, and 8x initial rise times to d3 folder
    for item in double_file_array:
        file_name1 = str(data_path / 'double_spe' / 'rt_1' / delay_folder / 'D2--waveforms--%s.txt') % item
        file_name2 = str(data_path / 'double_spe' / 'rt_2' / delay_folder / 'D2--waveforms--%s.txt') % item
        file_name4 = str(data_path / 'double_spe' / 'rt_4' / delay_folder / 'D2--waveforms--%s.txt') % item
        file_name8 = str(data_path / 'double_spe' / 'rt_8' / delay_folder / 'D2--waveforms--%s.txt') % item
        save_name1 = str(dest_path / 'double_spe' / 'rt_1' / delay_folder / 'raw' / 'D3--waveforms--%s.txt') % item
        save_name2 = str(dest_path / 'double_spe' / 'rt_2' / delay_folder / 'raw' / 'D3--waveforms--%s.txt') % item
        save_name4 = str(dest_path / 'double_spe' / 'rt_4' / delay_folder / 'raw' / 'D3--waveforms--%s.txt') % item
        save_name8 = str(dest_path / 'double_spe' / 'rt_8' / delay_folder / 'raw' / 'D3--waveforms--%s.txt') % item

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

    # Copies single spe waveforms with 1x, 2x, 4x, and 8x initial rise times to d3 folder
    for item in single_file_array:
        file_name1 = str(data_path / 'single_spe' / 'rt_1' / 'D2--waveforms--%s.txt') % item
        file_name2 = str(data_path / 'single_spe' / 'rt_2' / 'D2--waveforms--%s.txt') % item
        file_name4 = str(data_path / 'single_spe' / 'rt_4' / 'D2--waveforms--%s.txt') % item
        file_name8 = str(data_path / 'single_spe' / 'rt_8' / 'D2--waveforms--%s.txt') % item
        save_name1 = str(dest_path / 'single_spe' / 'rt_1' / 'raw' / 'D3--waveforms--%s.txt') % item
        save_name2 = str(dest_path / 'single_spe' / 'rt_2' / 'raw' / 'D3--waveforms--%s.txt') % item
        save_name4 = str(dest_path / 'single_spe' / 'rt_4' / 'raw' / 'D3--waveforms--%s.txt') % item
        save_name8 = str(dest_path / 'single_spe' / 'rt_8' / 'raw' / 'D3--waveforms--%s.txt') % item

        if os.path.isfile(file_name1):
            if os.path.isfile(save_name1):
                print('File #%s in rt_1 folder' % item)
            else:
                t, v, hdr = rw(file_name1, nhdr)
                ww(t, v, save_name1, hdr)
                print('File #%s in rt_1 folder' % item)

        if os.path.isfile(file_name2):
            if os.path.isfile(save_name2):
                print('File #%s in rt_2 folder' % item)
            else:
                t, v, hdr = rw(file_name2, nhdr)
                ww(t, v, save_name2, hdr)
                print('File #%s in rt_2 folder' % item)

        if os.path.isfile(file_name4):
            if os.path.isfile(save_name4):
                print('File #%s in rt_4 folder' % item)
            else:
                t, v, hdr = rw(file_name4, nhdr)
                ww(t, v, save_name4, hdr)
                print('File #%s in rt_4 folder' % item)

        if os.path.isfile(file_name8):
            if os.path.isfile(save_name8):
                print('File #%s in rt_8 folder' % item)
            else:
                t, v, hdr = rw(file_name8, nhdr)
                ww(t, v, save_name8, hdr)
                print('File #%s in rt_8 folder' % item)

    # Downsamples double spe waveforms using given fsps
    for item in double_file_array:
        file_name1 = str(dest_path / 'double_spe' / 'rt_1' / delay_folder / 'raw' / 'D3--waveforms--%s.txt') % item
        file_name2 = str(dest_path / 'double_spe' / 'rt_2' / delay_folder / 'raw' / 'D3--waveforms--%s.txt') % item
        file_name4 = str(dest_path / 'double_spe' / 'rt_4' / delay_folder / 'raw' / 'D3--waveforms--%s.txt') % item
        file_name8 = str(dest_path / 'double_spe' / 'rt_8' / delay_folder / 'raw' / 'D3--waveforms--%s.txt') % item
        save_name1 = str(dest_path / 'double_spe' / 'rt_1' / delay_folder /
                         str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') / 'D3--waveforms--%s.txt') % item
        save_name2 = str(dest_path / 'double_spe' / 'rt_2' / delay_folder /
                         str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') / 'D3--waveforms--%s.txt') % item
        save_name4 = str(dest_path / 'double_spe' / 'rt_4' / delay_folder /
                         str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') / 'D3--waveforms--%s.txt') % item
        save_name8 = str(dest_path / 'double_spe' / 'rt_8' / delay_folder /
                         str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') / 'D3--waveforms--%s.txt') % item

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

    # Copies downsampled single spe waveforms
    for item in single_file_array:
        file_name1 = str(dest_path / 'rt_1' / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%s.txt') % item
        file_name2 = str(dest_path / 'rt_2' / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%s.txt') % item
        file_name4 = str(dest_path / 'rt_4' / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%s.txt') % item
        file_name8 = str(dest_path / 'rt_8' / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%s.txt') % item
        save_name1 = str(dest_path / 'single_spe' / 'rt_1' / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%s.txt') % item
        save_name2 = str(dest_path / 'single_spe' / 'rt_2' / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%s.txt') % item
        save_name4 = str(dest_path / 'single_spe' / 'rt_4' / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%s.txt') % item
        save_name8 = str(dest_path / 'single_spe' / 'rt_8' / str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%s.txt') % item

        if os.path.isfile(file_name1):
            if os.path.isfile(save_name1):
                print('File #%s in rt_1 folder' % item)
            else:
                t, v, hdr = rw(file_name1, nhdr)
                ww(t, v, save_name1, hdr)
                print('File #%s in rt_1 folder' % item)

        if os.path.isfile(file_name2):
            if os.path.isfile(save_name2):
                print('File #%s in rt_2 folder' % item)
            else:
                t, v, hdr = rw(file_name2, nhdr)
                ww(t, v, save_name2, hdr)
                print('File #%s in rt_2 folder' % item)

        if os.path.isfile(file_name4):
            if os.path.isfile(save_name4):
                print('File #%s in rt_4 folder' % item)
            else:
                t, v, hdr = rw(file_name4, nhdr)
                ww(t, v, save_name4, hdr)
                print('File #%s in rt_4 folder' % item)

        if os.path.isfile(file_name8):
            if os.path.isfile(save_name8):
                print('File #%s in rt_8 folder' % item)
            else:
                t, v, hdr = rw(file_name8, nhdr)
                ww(t, v, save_name8, hdr)
                print('File #%s in rt_8 folder' % item)

    # Digitizes double spe waveforms using given noise
    for item in double_file_array:
        file_name1 = str(dest_path / 'double_spe' / 'rt_1' / delay_folder /
                         str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') / 'D3--waveforms--%s.txt') % item
        file_name2 = str(dest_path / 'double_spe' / 'rt_2' / delay_folder /
                         str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') / 'D3--waveforms--%s.txt') % item
        file_name4 = str(dest_path / 'double_spe' / 'rt_4' / delay_folder /
                         str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') / 'D3--waveforms--%s.txt') % item
        file_name8 = str(dest_path / 'double_spe' / 'rt_8' / delay_folder /
                         str('downsampled_' + str(int(fsps_new / 1e6)) + '_Msps') / 'D3--waveforms--%s.txt') % item
        save_name1 = str(dest_path / 'double_spe' / 'rt_1' / delay_folder /
                         str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') / 'D3--waveforms--%s.txt') % item
        save_name2 = str(dest_path / 'double_spe' / 'rt_2' / delay_folder /
                         str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') / 'D3--waveforms--%s.txt') % item
        save_name4 = str(dest_path / 'double_spe' / 'rt_4' / delay_folder /
                         str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') / 'D3--waveforms--%s.txt') % item
        save_name8 = str(dest_path / 'double_spe' / 'rt_8' / delay_folder /
                         str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') / 'D3--waveforms--%s.txt') % item

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

    # Copies digitized single spe waveforms
    for item in single_file_array:
        file_name1 = str(dest_path / 'rt_1' / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%s.txt') % item
        file_name2 = str(dest_path / 'rt_2' / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%s.txt') % item
        file_name4 = str(dest_path / 'rt_4' / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%s.txt') % item
        file_name8 = str(dest_path / 'rt_8' / str('digitized_' + str(int(fsps_new / 1e6)) + '_Msps') /
                         'D3--waveforms--%s.txt') % item
        save_name1 = str(dest_path / 'single_spe' / 'rt_1' / str('digitized_' + str(int(fsps_new / 1e6)) +
                                                                 '_Msps') / 'D3--waveforms--%s.txt') % item
        save_name2 = str(dest_path / 'single_spe' / 'rt_2' / str('digitized_' + str(int(fsps_new / 1e6)) +
                                                                 '_Msps') / 'D3--waveforms--%s.txt') % item
        save_name4 = str(dest_path / 'single_spe' / 'rt_4' / str('digitized_' + str(int(fsps_new / 1e6)) +
                                                                 '_Msps') / 'D3--waveforms--%s.txt') % item
        save_name8 = str(dest_path / 'single_spe' / 'rt_8' / str('digitized_' + str(int(fsps_new / 1e6)) +
                                                                 '_Msps') / 'D3--waveforms--%s.txt') % item

        if os.path.isfile(file_name1):
            if os.path.isfile(save_name1):
                print('File #%s in rt_1 folder' % item)
            else:
                t, v, hdr = rw(file_name1, nhdr)
                ww(t, v, save_name1, hdr)
                print('File #%s in rt_1 folder' % item)

        if os.path.isfile(file_name2):
            if os.path.isfile(save_name2):
                print('File #%s in rt_2 folder' % item)
            else:
                t, v, hdr = rw(file_name2, nhdr)
                ww(t, v, save_name2, hdr)
                print('File #%s in rt_2 folder' % item)

        if os.path.isfile(file_name4):
            if os.path.isfile(save_name4):
                print('File #%s in rt_4 folder' % item)
            else:
                t, v, hdr = rw(file_name4, nhdr)
                ww(t, v, save_name4, hdr)
                print('File #%s in rt_4 folder' % item)

        if os.path.isfile(file_name8):
            if os.path.isfile(save_name8):
                print('File #%s in rt_8 folder' % item)
            else:
                t, v, hdr = rw(file_name8, nhdr)
                ww(t, v, save_name8, hdr)
                print('File #%s in rt_8 folder' % item)

    # Creates double spe arrays for charge, amplitude, and FWHM
    charge_array, amplitude_array, fwhm_array = make_arrays(double_file_array, 'rt_1', delay_folder, dest_path,
                                                            nhdr, r, fsps_new)
    charge_array_2, amplitude_array_2, fwhm_array_2 = make_arrays(double_file_array, 'rt_2', delay_folder,
                                                                  dest_path, nhdr, r, fsps_new)
    charge_array_4, amplitude_array_4, fwhm_array_4 = make_arrays(double_file_array, 'rt_4', delay_folder,
                                                                  dest_path, nhdr, r, fsps_new)
    charge_array_8, amplitude_array_8, fwhm_array_8 = make_arrays(double_file_array, 'rt_8', delay_folder,
                                                                  dest_path, nhdr, r, fsps_new)

    # Creates single spe arrays for charge, amplitude, and FWHM
    charge_array_s, amplitude_array_s, fwhm_array_s = make_arrays_s(single_file_array, dest_path, 'rt_1', nhdr, r,
                                                                    fsps_new)
    charge_array_2_s, amplitude_array_2_s, fwhm_array_2_s = make_arrays_s(single_file_array, dest_path, 'rt_2', nhdr, r,
                                                                          fsps_new)
    charge_array_4_s, amplitude_array_4_s, fwhm_array_4_s = make_arrays_s(single_file_array, dest_path, 'rt_4', nhdr, r,
                                                                          fsps_new)
    charge_array_8_s, amplitude_array_8_s, fwhm_array_8_s = make_arrays_s(single_file_array, dest_path, 'rt_8', nhdr, r,
                                                                          fsps_new)

    # Creates double spe histograms for charge, amplitude, and FWHM
    print('Creating double spe histograms...')
    plot_histogram(charge_array, dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_double_rt1_' +
                   str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(charge_array_2, dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_double_rt2_' +
                   str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(charge_array_4, dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_double_rt4_' +
                   str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(charge_array_8, dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_double_rt8_' +
                   str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(amplitude_array, dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_double_rt1_' +
                   str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(amplitude_array_2, dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_double_rt2_' +
                   str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(amplitude_array_4, dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_double_rt4_' +
                   str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(amplitude_array_8, dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_double_rt8_' +
                   str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(fwhm_array, dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_double_rt1_' + str(delay_folder) + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(fwhm_array_2, dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_double_rt2_' + str(delay_folder) + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(fwhm_array_4, dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_double_rt4_' + str(delay_folder) + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(fwhm_array_8, dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_double_rt8_' + str(delay_folder) + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)

    # Creates single spe histograms for charge, amplitude, and FWHM
    print('Creating single spe histograms...')
    plot_histogram(charge_array_s, dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_single_rt1' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(charge_array_2_s, dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_single_rt2' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(charge_array_4_s, dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_single_rt4' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(charge_array_8_s, dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_single_rt8' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(amplitude_array_s, dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_single_rt1' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(amplitude_array_2_s, dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_single_rt2' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(amplitude_array_4_s, dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_single_rt4' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(amplitude_array_8_s, dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_single_rt8' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(fwhm_array_s, dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_single_rt1' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(fwhm_array_2_s, dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_single_rt2' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(fwhm_array_4_s, dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_single_rt4' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_histogram(fwhm_array_8_s, dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_single_rt8' + '_' +
                   str(int(fsps_new / 1e6)) + '_Msps', fsps_new)

    # Creates double & single spe histograms on same plot
    print('Creating histograms...')
    plot_double_hist(dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_single_rt1', 'charge_double_rt1_' +
                     str(delay_folder), 'charge_rt1_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps',
                     fsps_new)
    plot_double_hist(dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_single_rt2', 'charge_double_rt2_' +
                     str(delay_folder), 'charge_rt2_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps',
                     fsps_new)
    plot_double_hist(dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_single_rt4', 'charge_double_rt4_' +
                     str(delay_folder), 'charge_rt4_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps',
                     fsps_new)
    plot_double_hist(dest_path, 75, 'Charge', 'Charge', 's*bit/ohm', 'charge_single_rt8', 'charge_double_rt8_' +
                     str(delay_folder), 'charge_rt8_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps',
                     fsps_new)
    plot_double_hist(dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_single_rt1', 'amp_double_rt1_' +
                     str(delay_folder), 'amp_rt1_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps',
                     fsps_new)
    plot_double_hist(dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_single_rt2', 'amp_double_rt2_'
                     + str(delay_folder), 'amp_rt2_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps',
                     fsps_new)
    plot_double_hist(dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_single_rt4', 'amp_double_rt4_'
                     + str(delay_folder), 'amp_rt4_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps',
                     fsps_new)
    plot_double_hist(dest_path, 75, 'Amplitude', 'Amplitude', 'bits', 'amp_single_rt8', 'amp_double_rt8_'
                     + str(delay_folder), 'amp_rt8_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps',
                     fsps_new)
    plot_double_hist(dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_single_rt1', 'fwhm_double_rt1_' + str(delay_folder),
                     'fwhm_rt1_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_double_hist(dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_single_rt2', 'fwhm_double_rt2_' + str(delay_folder),
                     'fwhm_rt2_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_double_hist(dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_single_rt4', 'fwhm_double_rt4_' + str(delay_folder),
                     'fwhm_rt4_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)
    plot_double_hist(dest_path, 75, 'Time', 'FWHM', 's', 'fwhm_single_rt8', 'fwhm_double_rt8_' + str(delay_folder),
                     'fwhm_rt8_' + str(delay_folder) + '_' + str(int(fsps_new / 1e6)) + '_Msps', fsps_new)


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
