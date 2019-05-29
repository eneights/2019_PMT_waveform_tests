from scipy import signal
from functions import *


# Separates files into spe, non-spe, and maybe spe
def p1_sort(file_num, nhdr, fsps, fc, numtaps, data_path, save_path, baseline):
    wc = 2. * np.pi * fc / fsps     # Discrete radial frequency
    lowpass = signal.firwin(numtaps, cutoff=wc/np.pi, window='blackman')    # Blackman windowed lowpass filter

    file_name = str(data_path / 'C2--waveforms--%05d.txt') % file_num
    spe_name = str(save_path / 'd1/d1_raw/D1--waveforms--%05d.txt') % file_num
    spe_not_there = str(save_path / 'd1/not_spe/D1--not_spe--%05d.txt') % file_num
    spe_unsure = str(save_path / 'd1/unsure_if_spe/D1--unsure--%05d.txt') % file_num

    if os.path.isfile(spe_name):
        pass
    elif os.path.isfile(spe_not_there):
        pass
    elif os.path.isfile(spe_unsure):
        pass
    else:
        t, v, hdr = rw(file_name, nhdr)

        v1 = signal.filtfilt(lowpass, 1.0, v - baseline)
        v2 = v1[numtaps:len(v1)-1]
        t2 = t[numtaps:len(v1)-1]

        v_flip = -1 * v2
        peaks, _ = signal.find_peaks(v_flip, 0.001)
        v_peaks = v2[peaks]
        t_peaks = t2[peaks]
        check_peaks, _ = signal.find_peaks(v_flip, [0.001, 0.0025])
        v_check = v2[check_peaks]

        # If no peaks larger than 0.001 V, no spe
        if len(peaks) == 0:
            ww(t2, v2, spe_not_there, hdr)
            print("Length of /d1_raw/:", len(os.listdir(str(save_path / 'd1/d1_raw/'))))

        # If one peak larger than 0.001 V and it is larger than 0.002 V, spe
        elif len(peaks) == 1 and min(v2[370:1370]) < -0.002:
            ww(t2, v2, spe_name, hdr)
            print("Length of /d1_raw/:", len(os.listdir(str(save_path / 'd1/d1_raw/'))))

        # If 2 or more peaks larger than 0.001 V, peak is larger than 0.005 V, and all other peaks are smaller than
        # 0.0025, spe
        elif len(peaks) >= 2 and min(v2[370:1370]) < -0.005 and len(peaks) - 1 == len(v_check):
            ww(t2, v2, spe_name, hdr)
            print("Length of /d1_raw/:", len(os.listdir(str(save_path / 'd1/d1_raw/'))))

        # Otherwise, plots waveform for user to sort manually
        else:
            plt.figure()
            plt.plot(t, v, 'b')
            plt.plot(t2, v2 + baseline, 'r', linewidth=2.5)
            plt.plot(t_peaks, v_peaks + baseline, 'x', color='cyan')
            plt.title('File #%05d' % file_num)
            plt.xlabel('Time (s)')
            plt.ylabel('Voltage (V)')
            plt.grid(True)
            print('Displaying file #%05d' % file_num)
            plt.show(block=False)

            spe_check = 'pre-loop initialization'
            while spe_check != 'y' and spe_check != 'n' and spe_check != 'u':
                spe_check = input('Is there a single visible SPE? "y", "n", or "u"\n')
            if spe_check == 'y':
                ww(t2, v2, spe_name, hdr)
            elif spe_check == 'n':
                ww(t2, v2, spe_not_there, hdr)
            elif spe_check == 'u':
                ww(t2, v2, spe_unsure, hdr)
            print('file #%05d: Done' % file_num)
            print("Length of /d1_raw/:", len(os.listdir(str(save_path / 'd1/d1_raw/'))))
            plt.close()

    return


if __name__ == '__main__':
    data = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/full_bdw_no_nf/d0')
    save = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/full_bdw_no_nf')
    import argparse
    parser = argparse.ArgumentParser(prog="p1 sort", description="Sorting through raw data to find good SPEs")
    parser.add_argument("--file_num", type=int, help='file number to begin at', default=00000)
    parser.add_argument("--nhdr", type=int, help='number of header lines to skip in the raw file', default=5)
    parser.add_argument("--fsps", type=float, help='samples per second (Hz)', default=20000000000.)
    parser.add_argument("--fc", type=float, help='filter cutoff frequency (Hz)', default=250000000.)
    parser.add_argument("--numtaps", type=int, help='filter order + 1', default=51)
    parser.add_argument("--data_path", type=str, help='folder to read from', default=data)
    parser.add_argument("--save_path", type=str, help='folder to save to', default=save)
    parser.add_argument("--baseline", type=float, help='baseline of data set (V)', default=0)
    args = parser.parse_args()

    p1_sort(args.file_num, args.nhdr, args.fsps, args.fc, args.numtaps, args.data_path, args.save_path, args.baseline)
