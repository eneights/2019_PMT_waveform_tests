import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from read_waveform import read_waveform as rw
from write_waveform import write_waveform as ww


def subtract_time(file_num, nhdr, data_path, save_path):
    file_name = 'D1--waveforms--%05d.txt' % file_num

    if os.path.isfile(data_path / file_name):
        t, v, hdr = rw(data_path / file_name, nhdr)
        half_max = min(v) / 2
        differential = np.diff(v)
        difference_value = np.abs(v - half_max)
        for i in range(0, len(differential)):
            if differential[i] > 0:
                difference_value[i] = np.inf
        index = np.argmin(difference_value)
        half_max_time = t[index]
        t2 = t - half_max_time
        idx = np.where(t2 == 0)
        idx = int(idx[0])
        t2 = np.roll(t2, len(t2) - idx)
        if idx != 0:
            for i in range(len(t2) - idx, len(t2) - 1):
                t2[i] = t2[i - 1] + t2[i + 1] - t2[i]
            t2[len(t2) - 1] = t2[len(t2) - 2] - t2[len(t2) - 1]
        v = np.roll(v, len(v) - idx)
        ww(t2, v, save_path / file_name, hdr)
        print('Length of /d1_shifted/:', len(os.listdir(str(save_path))))
        half_max = half_max.item()
        return half_max


if __name__ == '__main__':
    data = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/full_bandwidth_no_noise_filter/d1/'
                r'd1_raw')
    save = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/full_bandwidth_no_noise_filter/d1/'
                r'd1_shifted')
    import argparse
    parser = argparse.ArgumentParser(prog="subtract time", description="shift waveform datafile to start at 50% max.")
    parser.add_argument("--file_num", type=int, help='file number to begin at', default=00000)
    parser.add_argument("--nhdr", type=int, help='number of header lines to skip in the raw file', default=5)
    parser.add_argument("--data_path", type=str, help='folder to read from', default=data)
    parser.add_argument("--save_path", type=str, help='folder to save to', default=save)
    args = parser.parse_args()

    half_max_v = subtract_time(args.file_num, args.nhdr, args.data_path, args.save_path)
