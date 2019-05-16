import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from read_waveform import read_waveform as rw
from write_waveform import write_waveform as ww


def subtract_time(file_num):
    nhdr = 5

    data_path = Path(r'/Users/Eliza/Documents/WATCHMAN/test_files/d1/d1_raw')
    # data_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/bandwidth/d1/d1_raw')
    save_path = Path(r'/Users/Eliza/Documents/WATCHMAN/test_files/d1/d1_shifted')
    # save_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/bandwidth/d1/d1_shifted')
    file_name = 'D1--waveforms--%05d.txt' % file_num

    if os.path.isfile(data_path / file_name):
        t, v, hdr = rw(data_path / file_name, nhdr)
        half_max = min(v) / 2
        tvals = np.linspace(4.64e-7, 6.64e-7, int(2e6))
        vvals = np.interp(tvals, t, v)
        differential = np.diff(vvals)
        difference_value = np.abs(vvals - half_max)
        for i in range(0, len(differential)):
            if differential[i] > 0:
                difference_value[i] = np.inf
        index = np.argmin(difference_value)
        half_max_time = tvals[index]
        t2 = t - half_max_time
        ww(t2, v, save_path / file_name, hdr)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="subtract time", description="subtract time interval from waveform datafile.")
    parser.add_argument("--file_num", type=int, help='file number to begin at', default=00000)
    args = parser.parse_args()

    subtract_time(args.file_num)
