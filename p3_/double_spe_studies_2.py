from functions import *
import math


def double_spe_studies_2(date, filter_band):
    gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
    save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (date, filter_band))
    dest_path = Path(save_path / 'd3')

    array_single = np.array([])
    array_5x = np.array([])
    array_1x = np.array([])
    array_15x = np.array([])
    array_2x = np.array([])
    array_25x = np.array([])
    array_3x = np.array([])
    array_single_amp = np.array([])
    array_single_charge = np.array([])
    array_double_amp = np.array([])
    array_double_charge = np.array([])

    spes_as_mpes_array = np.array([])
    mpes_as_spes_5x_array = np.array([])
    mpes_as_spes_1x_array = np.array([])
    mpes_as_spes_15x_array = np.array([])
    mpes_as_spes_2x_array = np.array([])
    mpes_as_spes_25x_array = np.array([])
    mpes_as_spes_3x_array = np.array([])
    fwhm_cutoff_array = np.array([])
    mpes_as_spes_array = np.array([])
    amp_cutoff_array = np.array([])
    charge_cutoff_array = np.array([])
    spes_as_mpes_array_3 = np.array([])
    spes_as_mpes_array_4 = np.array([])
    mpes_as_spes_array_2 = np.array([])
    mpes_as_spes_array_3 = np.array([])

    filename1 = 'fwhm_single_rt1'
    filename2 = 'fwhm_double_rt1_0.5x_rt'
    filename3 = 'fwhm_double_rt1_1x_rt'
    filename4 = 'fwhm_double_rt1_1.5x_rt'
    filename5 = 'fwhm_double_rt1_2x_rt'
    filename6 = 'fwhm_double_rt1_2.5x_rt'
    filename7 = 'fwhm_double_rt1_3x_rt'
    filename8 = 'amp_single_rt1'
    filename9 = 'amp_double_rt1_no_delay'
    filename10 = 'charge_single_rt1'
    filename11 = 'charge_double_rt1_no_delay'

    myfile1 = open(dest_path / 'hist_data' / str(filename1 + '.txt'), 'r')      # Opens histogram file
    for line in myfile1:
        line = line.strip()
        line = float(line)
        array_single = np.append(array_single, line)        # Reads values & saves in an array
    myfile1.close()                                         # Closes histogram file

    myfile2 = open(dest_path / 'hist_data' / str(filename2 + '.txt'), 'r')      # Opens histogram file
    for line in myfile2:
        line = line.strip()
        line = float(line)
        array_5x = np.append(array_5x, line)                # Reads values & saves in an array
    myfile2.close()                                         # Closes histogram file

    myfile3 = open(dest_path / 'hist_data' / str(filename3 + '.txt'), 'r')      # Opens histogram file
    for line in myfile3:
        line = line.strip()
        line = float(line)
        array_1x = np.append(array_1x, line)                # Reads values & saves in an array
    myfile3.close()                                         # Closes histogram file

    myfile4 = open(dest_path / 'hist_data' / str(filename4 + '.txt'), 'r')      # Opens histogram file
    for line in myfile4:
        line = line.strip()
        line = float(line)
        array_15x = np.append(array_15x, line)              # Reads values & saves in an array
    myfile4.close()                                         # Closes histogram file

    myfile5 = open(dest_path / 'hist_data' / str(filename5 + '.txt'), 'r')      # Opens histogram file
    for line in myfile5:
        line = line.strip()
        line = float(line)
        array_2x = np.append(array_2x, line)                # Reads values & saves in an array
    myfile5.close()                                         # Closes histogram file

    myfile6 = open(dest_path / 'hist_data' / str(filename6 + '.txt'), 'r')      # Opens histogram file
    for line in myfile6:
        line = line.strip()
        line = float(line)
        array_25x = np.append(array_25x, line)              # Reads values & saves in an array
    myfile6.close()                                         # Closes histogram file

    myfile7 = open(dest_path / 'hist_data' / str(filename7 + '.txt'), 'r')      # Opens histogram file
    for line in myfile7:
        line = line.strip()
        line = float(line)
        array_3x = np.append(array_3x, line)                # Reads values & saves in an array
    myfile7.close()                                         # Closes histogram file

    myfile8 = open(dest_path / 'hist_data' / str(filename8 + '.txt'), 'r')      # Opens histogram file
    for line in myfile8:
        line = line.strip()
        line = float(line)
        array_single_amp = np.append(array_single_amp, line)    # Reads values & saves in an array
    myfile8.close()                                             # Closes histogram file

    myfile9 = open(dest_path / 'hist_data' / str(filename9 + '.txt'), 'r')  # Opens histogram file
    for line in myfile9:
        line = line.strip()
        line = float(line)
        array_double_amp = np.append(array_double_amp, line)    # Reads values & saves in an array
    myfile9.close()                                             # Closes histogram file

    myfile10 = open(dest_path / 'hist_data' / str(filename10 + '.txt'), 'r')  # Opens histogram file
    for line in myfile10:
        line = line.strip()
        line = float(line)
        array_single_charge = np.append(array_single_charge, line)  # Reads values & saves in an array
    myfile10.close()                                                # Closes histogram file

    myfile11 = open(dest_path / 'hist_data' / str(filename11 + '.txt'), 'r')  # Opens histogram file
    for line in myfile11:
        line = line.strip()
        line = float(line)
        array_double_charge = np.append(array_double_charge, line)  # Reads values & saves in an array
    myfile11.close()                                                # Closes histogram file

    mean_single = np.mean(array_single).item()
    sd_single = np.std(array_single).item()
    mean_5x = np.mean(array_5x).item()
    sd_5x = np.std(array_5x).item()
    mean_1x = np.mean(array_1x).item()
    sd_1x = np.std(array_1x).item()
    mean_15x = np.mean(array_15x).item()
    sd_15x = np.std(array_15x).item()
    mean_2x = np.mean(array_2x).item()
    sd_2x = np.std(array_2x).item()
    mean_25x = np.mean(array_25x).item()
    sd_25x = np.std(array_25x).item()
    mean_3x = np.mean(array_3x).item()
    sd_3x = np.std(array_3x).item()
    mean_single_amp = np.mean(array_single_amp).item()
    sd_single_amp = np.std(array_single_amp).item()
    mean_double_amp = np.mean(array_double_amp).item()
    sd_double_amp = np.std(array_double_amp).item()
    mean_single_charge = np.mean(array_single_charge).item()
    sd_single_charge = np.std(array_single_charge).item()
    mean_double_charge = np.mean(array_double_charge).item()
    sd_double_charge = np.std(array_double_charge).item()

    print('Calculating FWHM cutoff...')

    for i in range(85, 150):
        x = i * 10**-10
        fwhm_cutoff_array = np.append(fwhm_cutoff_array, x)
        spes_as_mpes = 100 * (1 + ((1 / 2) * (-2 + math.erfc((x - mean_single) / (sd_single * math.sqrt(2))))))
        mpes_as_spes_5x = 100 * ((1 / 2) * (2 - math.erfc((x - mean_5x) / (sd_5x * math.sqrt(2)))))
        mpes_as_spes_1x = 100 * ((1 / 2) * (2 - math.erfc((x - mean_1x) / (sd_1x * math.sqrt(2)))))
        mpes_as_spes_15x = 100 * ((1 / 2) * (2 - math.erfc((x - mean_15x) / (sd_15x * math.sqrt(2)))))
        mpes_as_spes_2x = 100 * ((1 / 2) * (2 - math.erfc((x - mean_2x) / (sd_2x * math.sqrt(2)))))
        mpes_as_spes_25x = 100 * ((1 / 2) * (2 - math.erfc((x - mean_25x) / (sd_25x * math.sqrt(2)))))
        mpes_as_spes_3x = 100 * ((1 / 2) * (2 - math.erfc((x - mean_3x) / (sd_3x * math.sqrt(2)))))
        spes_as_mpes_array = np.append(spes_as_mpes_array, spes_as_mpes)
        mpes_as_spes_5x_array = np.append(mpes_as_spes_5x_array, mpes_as_spes_5x)
        mpes_as_spes_1x_array = np.append(mpes_as_spes_1x_array, mpes_as_spes_1x)
        mpes_as_spes_15x_array = np.append(mpes_as_spes_15x_array, mpes_as_spes_15x)
        mpes_as_spes_2x_array = np.append(mpes_as_spes_2x_array, mpes_as_spes_2x)
        mpes_as_spes_25x_array = np.append(mpes_as_spes_25x_array, mpes_as_spes_25x)
        mpes_as_spes_3x_array = np.append(mpes_as_spes_3x_array, mpes_as_spes_3x)

    fwhm_cutoff_array_2 = np.linspace(8.5e-9, 1.5e-8, 1000)
    spes_as_mpes_array_2 = np.interp(fwhm_cutoff_array_2, fwhm_cutoff_array, spes_as_mpes_array)
    mpes_as_spes_5x_array_2 = np.interp(fwhm_cutoff_array_2, fwhm_cutoff_array, mpes_as_spes_5x_array)
    mpes_as_spes_1x_array_2 = np.interp(fwhm_cutoff_array_2, fwhm_cutoff_array, mpes_as_spes_1x_array)
    mpes_as_spes_15x_array_2 = np.interp(fwhm_cutoff_array_2, fwhm_cutoff_array, mpes_as_spes_15x_array)
    mpes_as_spes_2x_array_2 = np.interp(fwhm_cutoff_array_2, fwhm_cutoff_array, mpes_as_spes_2x_array)
    mpes_as_spes_25x_array_2 = np.interp(fwhm_cutoff_array_2, fwhm_cutoff_array, mpes_as_spes_25x_array)
    mpes_as_spes_3x_array_2 = np.interp(fwhm_cutoff_array_2, fwhm_cutoff_array, mpes_as_spes_3x_array)

    idx = np.argmin(np.abs(spes_as_mpes_array_2 - 1))
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_5x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_1x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_15x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_2x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_25x_array_2[idx])
    mpes_as_spes_array = np.append(mpes_as_spes_array, mpes_as_spes_3x_array_2[idx])

    delay_array = np.array([1.52e-9, 3.04e-9, 4.56e-9, 6.08e-9, 7.6e-9, 9.12e-9])
    fwhm_cutoff = str(float(format(fwhm_cutoff_array_2[idx], '.2e')))

    plt.scatter(delay_array, mpes_as_spes_array)
    plt.plot(delay_array, mpes_as_spes_array)
    plt.xlim(1.3e-9, 9.4e-9)
    plt.ylim(-5, 100)
    plt.xlabel('Delay (s)')
    plt.ylabel('% MPES Misidentified as SPEs')
    plt.title('False SPEs\nFWHM Cutoff = ' + fwhm_cutoff + ' s')
    for i in range(len(mpes_as_spes_array) - 1):
        pt = str(float(format(mpes_as_spes_array[i], '.1e')))
        plt.annotate(pt + '%', (delay_array[i], mpes_as_spes_array[i] + 1))
    plt.savefig(dest_path / 'plots' / 'false_spes_fwhm.png', dpi=360)
    plt.close()

    for i in range(100, 500):
        x = i
        amp_cutoff_array = np.append(amp_cutoff_array, x)
        spes_as_mpes = 100 * (1 + ((1 / 2) * (-2 + math.erfc((x - mean_single_amp) / (sd_single_amp * math.sqrt(2))))))
        mpes_as_spes = 100 * ((1 / 2) * (2 - math.erfc((x - mean_double_amp) / (sd_double_amp * math.sqrt(2)))))
        spes_as_mpes_array_3 = np.append(spes_as_mpes_array_3, spes_as_mpes)
        mpes_as_spes_array_2 = np.append(mpes_as_spes_array_2, mpes_as_spes)

    for i in range(20, 120):
        x = i * 10**-9
        charge_cutoff_array = np.append(charge_cutoff_array, x)
        spes_as_mpes = 100 * (1 + ((1 / 2) * (-2 + math.erfc((x - mean_single_charge) / (sd_single_charge *
                                                                                         math.sqrt(2))))))
        mpes_as_spes = 100 * ((1 / 2) * (2 - math.erfc((x - mean_double_charge) / (sd_double_charge * math.sqrt(2)))))
        spes_as_mpes_array_4 = np.append(spes_as_mpes_array_4, spes_as_mpes)
        mpes_as_spes_array_3 = np.append(mpes_as_spes_array_3, mpes_as_spes)

    plt.plot(amp_cutoff_array, spes_as_mpes_array_3)
    plt.ylim(-5, 100)
    plt.xlabel('Amplitude Cutoff (bits)')
    plt.ylabel('% SPES Misidentified as MPEs')
    plt.title('False MPEs')
    plt.savefig(dest_path / 'plots' / 'false_mpes_amp.png', dpi=360)
    plt.close()

    plt.plot(amp_cutoff_array, mpes_as_spes_array_2)
    plt.ylim(-5, 100)
    plt.xlabel('Amplitude Cutoff (bits)')
    plt.ylabel('% MPES Misidentified as SPEs')
    plt.title('False SPEs')
    plt.savefig(dest_path / 'plots' / 'false_spes_amp.png', dpi=360)
    plt.close()

    plt.plot(charge_cutoff_array, spes_as_mpes_array_4)
    plt.ylim(-5, 100)
    plt.xlabel('Charge Cutoff (s*bit/ohm)')
    plt.ylabel('% SPES Misidentified as MPEs')
    plt.title('False MPEs')
    plt.savefig(dest_path / 'plots' / 'false_mpes_charge.png', dpi=360)
    plt.close()

    plt.plot(charge_cutoff_array, mpes_as_spes_array_3)
    plt.ylim(-5, 100)
    plt.xlabel('Charge Cutoff (s*bit/ohm)')
    plt.ylabel('% MPES Misidentified as SPEs')
    plt.title('False SPEs')
    plt.savefig(dest_path / 'plots' / 'false_spes_charge.png', dpi=360)
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="double_spe_studies_2", description="Analyzing double spe data")
    parser.add_argument("--date", type=int, help='date of data acquisition (default=20190513)', default=20190513)
    parser.add_argument("--fil_band", type=str, help='folder name for data (default=full_bdw_no_nf)',
                        default='full_bdw_no_nf')
    args = parser.parse_args()

    double_spe_studies_2(args.date, args.fil_band)
