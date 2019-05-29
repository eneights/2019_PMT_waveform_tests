from functions import *


def p1b(start, end, dest_path, nhdr):
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
    file_array = np.array([])
    file_array2 = np.array([])

    for i in range(start, end + 1):
        file_name = str(dest_path / 'calculations' / 'D1--waveforms--%05d.txt') % i
        if os.path.isfile(file_name):
            myfile = open(file_name, 'r')
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

            if charge <= 0 or charge >= 2e-12:
                file_array = np.append(file_array, i)
            elif amplitude <= 0 or amplitude >= 0.014:
                file_array = np.append(file_array, i)
            elif fwhm <= 7e-9 or fwhm >= 9e-9:
                file_array = np.append(file_array, i)
            elif rise1090 <= 2.6e-9 or rise1090 >= 4e-9:
                file_array = np.append(file_array, i)
            elif rise2080 <= 1.7e-9 or rise2080 >= 3e-9:
                file_array = np.append(file_array, i)
            elif fall1090 <= 1.4e-8 or fall1090 >= 2e-8:
                file_array = np.append(file_array, i)
            elif fall2080 <= 9e-9 or fall2080 >= 1.25e-8:
                file_array = np.append(file_array, i)
            elif time10 <= -4e-9 or time10 >= 0:
                file_array = np.append(file_array, i)
            elif time20 <= -2.5e-9 or time20 >= 0:
                file_array = np.append(file_array, i)
            elif time80 <= 0 or time80 >= 1.5e-9:
                file_array = np.append(file_array, i)
            elif time90 <= 0 or time90 >= 2.5e-9:
                file_array = np.append(file_array, i)

    h = plt.hist2d(fall1090_array, charge_array, bins=30)
    plt.xlabel('10-90 Fall Time (s)')
    plt.ylabel('Charge (C)')
    plt.colorbar(h[3])
    plt.show()

    '''for i in range(start, end + 1):
        if i in file_array:
            file_name = str(dest_path / 'd1_shifted' / 'D1--waveforms--%05d.txt') % i
            t, v, hdr = rw(file_name, nhdr)
            print('Displaying file #%05d' % i)
            plt.figure()
            plt.plot(t, v)
            plt.title('File #%05d' % i)
            plt.xlabel('Time (s)')
            plt.ylabel('Voltage (V)')
            plt.show()
            spe_check = 'pre-loop initialization'
            while spe_check != 'y' and spe_check != 'n' and spe_check != 'u':
                spe_check = input('Is this a normal single SPE? "y", "n", or "u"\n')
            if spe_check == 'y':
                pass
            elif spe_check == 'n':
                print('Removing file #%05d' % i)
                location1 = file_name
                location2 = str(dest_path / 'd1_raw' / 'D1--waveforms--%05d.txt') % i
                location3 = str(dest_path / 'calculations' / 'D1--waveforms--%05d.txt') % i
                location4 = str(dest_path / 'not_spe' / 'D1--not_spe--%05d.txt') % i
                os.remove(location1)
                os.remove(location2)
                os.remove(location3)
                ww(t, v, location4, hdr)
            elif spe_check == 'u':
                file_array2 = np.append(file_array2, i)
            plt.close()
            
    return file_array2'''


if __name__ == '__main__':
    data = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/full_bdw_no_nf/d1')
    import argparse
    parser = argparse.ArgumentParser(prog="p1b", description="Removing outliers from data set")
    parser.add_argument("--start", type=int, help='file number to begin at', default=00000)
    parser.add_argument("--end", type=int, help='file number to end at', default=99999)
    parser.add_argument("--nhdr", type=int, help='number of header lines to skip', default=5)
    parser.add_argument("--dest_path", type=str, help='folder to read from', default=data)
    args = parser.parse_args()

    # unsure_files = p1b(args.start, args.end, args.dest_path, args.nhdr)
    p1b(args.start, args.end, args.dest_path, args.nhdr)
