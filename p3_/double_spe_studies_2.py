from functions import *
import math


def double_spe_studies_2(date, filter_band, fsps_new, shaping):
    gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
    save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (date, filter_band))
    dest_path = Path(save_path / 'd3')

    filename1 = 'fwhm_single_' + shaping
    filename2 = 'fwhm_double_' + shaping + '_0.5x_rt'
    filename3 = 'fwhm_double_' + shaping + '_1x_rt'
    filename4 = 'fwhm_double_' + shaping + '_1.5x_rt'
    filename5 = 'fwhm_double_' + shaping + '_2x_rt'
    filename6 = 'fwhm_double_' + shaping + '_2.5x_rt'
    filename7 = 'fwhm_double_' + shaping + '_3x_rt'
    filename8 = 'amp_single_' + shaping
    filename9 = 'amp_double_' + shaping + '_no_delay'
    filename10 = 'charge_single_' + shaping
    filename11 = 'charge_double_' + shaping + '_no_delay'
    filename12 = 'charge_double_' + shaping + '_0.5x_rt'
    filename13 = 'charge_double_' + shaping + '_1x_rt'
    filename14 = 'charge_double_' + shaping + '_1.5x_rt'
    filename15 = 'charge_double_' + shaping + '_2x_rt'
    filename16 = 'charge_double_' + shaping + '_2.5x_rt'
    filename17 = 'charge_double_' + shaping + '_3x_rt'

    array_single_fwhm = read_hist_file(dest_path / 'hist_data', filename1, fsps_new)
    array_double_fwhm_5x = read_hist_file(dest_path / 'hist_data', filename2, fsps_new)
    array_double_fwhm_1x = read_hist_file(dest_path / 'hist_data', filename3, fsps_new)
    array_double_fwhm_15x = read_hist_file(dest_path / 'hist_data', filename4, fsps_new)
    array_double_fwhm_2x = read_hist_file(dest_path / 'hist_data', filename5, fsps_new)
    array_double_fwhm_25x = read_hist_file(dest_path / 'hist_data', filename6, fsps_new)
    array_double_fwhm_3x = read_hist_file(dest_path / 'hist_data', filename7, fsps_new)
    array_single_amp = read_hist_file(dest_path / 'hist_data', filename8, fsps_new)
    array_double_amp_nd = read_hist_file(dest_path / 'hist_data', filename9, fsps_new)
    array_single_charge = read_hist_file(dest_path / 'hist_data', filename10, fsps_new)
    array_double_charge_nd = read_hist_file(dest_path / 'hist_data', filename11, fsps_new)
    array_double_charge_5x = read_hist_file(dest_path / 'hist_data', filename12, fsps_new)
    array_double_charge_1x = read_hist_file(dest_path / 'hist_data', filename13, fsps_new)
    array_double_charge_15x = read_hist_file(dest_path / 'hist_data', filename14, fsps_new)
    array_double_charge_2x = read_hist_file(dest_path / 'hist_data', filename15, fsps_new)
    array_double_charge_25x = read_hist_file(dest_path / 'hist_data', filename16, fsps_new)
    array_double_charge_3x = read_hist_file(dest_path / 'hist_data', filename17, fsps_new)

    mean_single_fwhm = np.mean(array_single_fwhm)
    mean_double_fwhm_5x = np.mean(array_double_fwhm_5x)
    mean_double_fwhm_1x = np.mean(array_double_fwhm_1x)
    mean_double_fwhm_15x = np.mean(array_double_fwhm_15x)
    mean_double_fwhm_2x = np.mean(array_double_fwhm_2x)
    mean_double_fwhm_25x = np.mean(array_double_fwhm_25x)
    mean_double_fwhm_3x = np.mean(array_double_fwhm_3x)
    mean_single_amp = np.mean(array_single_amp)
    mean_double_amp_nd = np.mean(array_double_amp_nd)
    mean_single_charge = np.mean(array_single_charge)
    mean_double_charge_nd = np.mean(array_double_charge_nd)
    mean_double_charge_5x = np.mean(array_double_charge_5x)
    mean_double_charge_1x = np.mean(array_double_charge_1x)
    mean_double_charge_15x = np.mean(array_double_charge_15x)
    mean_double_charge_2x = np.mean(array_double_charge_2x)
    mean_double_charge_25x = np.mean(array_double_charge_25x)
    mean_double_charge_3x = np.mean(array_double_charge_3x)

    std_single_fwhm = np.std(array_single_fwhm)
    std_double_fwhm_5x = np.std(array_double_fwhm_5x)
    std_double_fwhm_1x = np.std(array_double_fwhm_1x)
    std_double_fwhm_15x = np.std(array_double_fwhm_15x)
    std_double_fwhm_2x = np.std(array_double_fwhm_2x)
    std_double_fwhm_25x = np.std(array_double_fwhm_25x)
    std_double_fwhm_3x = np.std(array_double_fwhm_3x)
    std_single_amp = np.std(array_single_amp)
    std_double_amp_nd = np.std(array_double_amp_nd)
    std_single_charge = np.std(array_single_charge)
    std_double_charge_nd = np.std(array_double_charge_nd)
    std_double_charge_5x = np.std(array_double_charge_5x)
    std_double_charge_1x = np.std(array_double_charge_1x)
    std_double_charge_15x = np.std(array_double_charge_15x)
    std_double_charge_2x = np.std(array_double_charge_2x)
    std_double_charge_25x = np.std(array_double_charge_25x)
    std_double_charge_3x = np.std(array_double_charge_3x)

    print('Making plots...')

    false_spes_vs_delay(85, 150, 10**-10, 'fwhm', 'FWHM', 's', fsps_new, mean_single_fwhm, mean_double_fwhm_5x,
                        mean_double_fwhm_1x, mean_double_fwhm_15x, mean_double_fwhm_2x, mean_double_fwhm_25x,
                        mean_double_fwhm_3x, std_single_fwhm, std_double_fwhm_5x, std_double_fwhm_1x,
                        std_double_fwhm_15x, std_double_fwhm_2x, std_double_fwhm_25x, std_double_fwhm_3x, dest_path)
    false_spes_vs_delay(20, 120, 10**-9, 'charge', 'Charge', 's*bit/ohm', fsps_new, mean_single_charge,
                        mean_double_charge_5x, mean_double_charge_1x, mean_double_charge_15x, mean_double_charge_2x,
                        mean_double_charge_25x, mean_double_charge_3x, std_single_charge, std_double_charge_5x,
                        std_double_charge_1x, std_double_charge_15x, std_double_charge_2x, std_double_charge_25x,
                        std_double_charge_3x, dest_path)

    false_spes_mpes(100, 500, 1, 'amp', 'Amplitude', 'bits', mean_single_amp, mean_double_amp_nd, std_single_amp,
                    std_double_amp_nd, fsps_new, dest_path)
    false_spes_mpes(20, 120, 10**-9, 'charge', 'Charge', 's*bit/ohm', mean_single_charge, mean_double_charge_nd,
                    std_single_charge, std_double_charge_nd, fsps_new, dest_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="double_spe_studies_2", description="Analyzing double spe data")
    parser.add_argument("--date", type=int, help='date of data acquisition (default=20190513)', default=20190513)
    parser.add_argument("--fil_band", type=str, help='folder name for data (default=full_bdw_no_nf)',
                        default='full_bdw_no_nf')
    parser.add_argument("--fsps_new", type=float, help='new samples per second (Hz) (default=500000000.)',
                        default=500000000.)
    parser.add_argument("--shaping", type=str, help='shaping amount (default=rt1)', default='rt1')
    args = parser.parse_args()

    double_spe_studies_2(args.date, args.fil_band, args.fsps_new, args.shaping)
