from functions import *


def p1b(start, end, dest_path, nhdr):
    charge_array_test = np.array([])
    fall1090_array_test = np.array([])
    charge_array = np.array([])
    amplitude_array = np.array([])
    fwhm_array = np.array([])
    rise1090_array = np.array([])
    rise2080_array = np.array([])
    fall1090_array = np.array([])
    fall2080_array = np.array([])
    time10_array = np.array([])
    time20_array = np.array([])
    time80_array = np.array([])
    time90_array = np.array([])
    imp_charge_array = np.array([])
    imp_amplitude_array = np.array([])
    imp_fwhm_array = np.array([])
    imp_rise1090_array = np.array([])
    imp_rise2080_array = np.array([])
    imp_fall1090_array = np.array([])
    imp_fall2080_array = np.array([])
    imp_time10_array = np.array([])
    imp_time20_array = np.array([])
    imp_time80_array = np.array([])
    imp_time90_array = np.array([])
    p1b_spe_array = np.array([])

    file_path_calc = Path(dest_path / 'calculations')
    file_path_shift = Path(dest_path / 'd1_shifted')
    file_path_shift_d1b = Path(dest_path / 'd1b_shifted')
    file_path_not_spe = Path(dest_path / 'd1b_not_spe')

    for i in range(start, end + 1):
        if os.path.isfile(file_path_calc / 'D1--waveforms--%05d.txt' % i):
            myfile = open(file_path_calc / 'D1--waveforms--%05d.txt' % i, 'r')
            csv_reader = csv.reader(myfile)
            file_array = np.array([])
            for row in csv_reader:
                file_array = np.append(file_array, float(row[1]))
            myfile.close()
            charge = file_array[2]
            amplitude = file_array[3]
            fwhm = file_array[4]
            rise1090 = file_array[5]
            rise2080 = file_array[6]
            fall1090 = file_array[7]
            fall2080 = file_array[8]
            time10 = file_array[9]
            time20 = file_array[10]
            time80 = file_array[11]
            time90 = file_array[12]

            charge_array_test = np.append(charge_array_test, charge)
            fall1090_array_test = np.append(fall1090_array_test, fall1090)

            if charge >= 2e-12:
                imp_charge_array = np.append(imp_charge_array, i)
            elif amplitude >= 0.014:
                imp_amplitude_array = np.append(imp_amplitude_array, i)
            elif fwhm <= 7e-9 or fwhm >= 9e-9:
                imp_fwhm_array = np.append(imp_fwhm_array, i)
            elif rise1090 <= 2.6e-9 or rise1090 >= 4e-9:
                imp_rise1090_array = np.append(imp_rise1090_array, i)
            elif rise2080 <= 1.7e-9 or rise2080 >= 3e-9:
                imp_rise2080_array = np.append(imp_rise2080_array, i)
            elif fall1090 <= 1.4e-8 or fall1090 >= 2.5e-8:
                imp_fall1090_array = np.append(imp_fall1090_array, i)
            elif fall2080 <= 9e-9 or fall2080 >= 1.25e-8:
                imp_fall2080_array = np.append(imp_fall2080_array, i)
            elif time10 <= -4e-9:
                imp_time10_array = np.append(imp_time10_array, i)
            elif time20 <= -2.5e-9:
                imp_time20_array = np.append(imp_time20_array, i)
            elif time80 >= 1.5e-9:
                imp_time80_array = np.append(imp_time80_array, i)
            elif time90 >= 2.5e-9:
                imp_time90_array = np.append(imp_time90_array, i)

    h = plt.hist2d(fall1090_array, charge_array, bins=40)
    plt.xlabel('10-90 Fall Time (s)')
    plt.ylabel('Charge (C)')
    plt.colorbar(h[3])
    plt.show()

    for i in range(start, end + 1):
        if i in (imp_charge_array or imp_amplitude_array or imp_fwhm_array or imp_rise1090_array or imp_rise2080_array
                 or imp_fall1090_array or imp_fall2080_array or imp_time10_array or imp_time20_array or imp_time80_array
                 or imp_time90_array):
            t, v, hdr = rw(file_path_shift / 'D1--waveforms--%05d.txt' % i, nhdr)
            myfile = open(file_path_calc / 'D1--waveforms--%05d.txt' % i, 'r')
            csv_reader = csv.reader(myfile)
            file_array = np.array([])
            for row in csv_reader:
                file_array = np.append(file_array, float(row[1]))
            myfile.close()
            charge = file_array[2]
            amplitude = file_array[3]
            fwhm = file_array[4]
            rise1090 = file_array[5]
            rise2080 = file_array[6]
            fall1090 = file_array[7]
            fall2080 = file_array[8]
            time10 = file_array[9]
            time20 = file_array[10]
            time80 = file_array[11]
            time90 = file_array[12]
            print('Displaying file #%05d' % i)
            print('charge = ', charge)
            print('amplitude = ', amplitude)
            print('FWHM = ', fwhm)
            print('10-90 rise time = ', rise1090)
            print('20-80 rise time = ', rise2080)
            print('10-90 fall time = ', fall1090)
            print('20-80 fall time = ', fall2080)
            print('10% jitter = ', time10)
            print('20% jitter = ', time20)
            print('80% jitter = ', time80)
            print('90% jitter = ', time90)
            plt.figure()
            plt.plot(t, v)
            plt.title('File #%05d' % i)
            plt.xlabel('Time (s)')
            plt.ylabel('Voltage (V)')
            plt.show()
            spe_check = 'pre-loop initialization'
            while spe_check != 'y' and spe_check != 'n' and spe_check != 'u':
                spe_check = input('Is this a normal SPE? "y" or "n"\n')
            if spe_check == 'y':
                if amplitude > 0.015 or charge > 2e-12:
                    print('File #%05d is not spe' % i)
                    ww(t, v, file_path_not_spe / 'D1--waveforms--%05d.txt' % i, hdr)
                else:
                    print('File #%05d is spe' % i)
                    ww(t, v, file_path_shift_d1b / 'D1--waveforms--%05d.txt' % i, hdr)
                    p1b_spe_array = np.append(p1b_spe_array, i)
            elif spe_check == 'n':
                print('File #%05d is not spe' % i)
                ww(t, v, file_path_not_spe / 'D1--waveforms--%05d.txt' % i, hdr)
            plt.close()
        else:
            if os.path.isfile(file_path_shift / 'D1--waveforms--%05d.txt' % i):
                t, v, hdr = rw(file_path_calc / 'D1--waveforms--%05d.txt' % i, nhdr)
                ww(t, v, file_path_shift_d1b / 'D1--waveforms--%05d.txt' % i, hdr)
                p1b_spe_array = np.append(p1b_spe_array, i)

    for i in range(start, end + 1):
        if i in p1b_spe_array:
            print("Reading calculations from shifted file #%05d" % i)
            myfile = open(file_path_calc / 'D1--waveforms--%05d.txt' % i, 'r')
            csv_reader = csv.reader(myfile)
            file_array = np.array([])
            for row in csv_reader:
                file_array = np.append(file_array, float(row[1]))
            myfile.close()
            charge = file_array[2]
            amplitude = file_array[3]
            fwhm = file_array[4]
            rise1090 = file_array[5]
            rise2080 = file_array[6]
            fall1090 = file_array[7]
            fall2080 = file_array[8]
            time10 = file_array[9]
            time20 = file_array[10]
            time80 = file_array[11]
            time90 = file_array[12]

            charge_array = np.append(charge_array, charge)
            amplitude_array = np.append(amplitude_array, amplitude)
            fwhm_array = np.append(fwhm_array, fwhm)
            rise1090_array = np.append(rise1090_array, rise1090)
            rise2080_array = np.append(rise2080_array, rise2080)
            fall1090_array = np.append(fall1090_array, fall1090)
            fall2080_array = np.append(fall2080_array, fall2080)
            time10_array = np.append(time10_array, time10)
            time20_array = np.append(time20_array, time20)
            time80_array = np.append(time80_array, time80)
            time90_array = np.append(time90_array, time90)

    plot_histogram(charge_array, dest_path, 100, 'Charge', 'Charge', 'C', 'charge_d1b')
    plot_histogram(amplitude_array, dest_path, 100, 'Voltage', 'Amplitude', 'V', 'amplitude_d1b')
    plot_histogram(fwhm_array, dest_path, 100, 'Time', 'FWHM', 's', 'fwhm_w_outliers')
    plot_histogram(rise1090_array, dest_path, 100, 'Time', '10-90 Rise Time', 's', 'rise1090_d1b')
    plot_histogram(rise2080_array, dest_path, 100, 'Time', '20-80 Rise Time', 's', 'rise2080_d1b')
    plot_histogram(fall1090_array, dest_path, 100, 'Time', '10-90 Fall Time', 's', 'fall1090_d1b')
    plot_histogram(fall2080_array, dest_path, 100, 'Time', '20-80 Fall Time', 's', 'fall2080_d1b')
    plot_histogram(time10_array, dest_path, 100, 'Time', '10% Jitter', 's', 'time10_d1b')
    plot_histogram(time20_array, dest_path, 100, 'Time', '20% Jitter', 's', 'time20_d1b')
    plot_histogram(time80_array, dest_path, 100, 'Time', '80% Jitter', 's', 'time80_d1b')
    plot_histogram(time90_array, dest_path, 100, 'Time', '90% Jitter', 's', 'time90_d1b')


if __name__ == '__main__':
    data = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/full_bdw_no_nf/d1')
    import argparse
    parser = argparse.ArgumentParser(prog="p1b", description="Removing outliers from data set")
    parser.add_argument("--start", type=int, help='file number to begin at', default=00000)
    parser.add_argument("--end", type=int, help='file number to end at', default=99999)
    parser.add_argument("--nhdr", type=int, help='number of header lines to skip', default=5)
    parser.add_argument("--dest_path", type=str, help='folder to read from', default=data)
    args = parser.parse_args()

    p1b(args.start, args.end, args.dest_path, args.nhdr)
