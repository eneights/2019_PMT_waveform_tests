from functions import *


# Plots a waveform for the user to view (does not save)
def plot_waveform(file_num, fil_band, version, folder):
    if version == 'd0':
        file_name = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/' + str(fil_band) +
                         r'/d0/C2--waveforms--%05d.txt' % file_num)
    else:
        file_name = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/' + str(fil_band) + '/d1/'
                         + str(folder) + r'/D1--waveforms--%05d.txt' % file_num)
    t, v, hdr = rw(file_name, 5)
    print("\nHeader:\n\n" + str(hdr))
    plt.plot(t, v)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="p1", description="Creating D1")
    parser.add_argument("--file_num", type=int, help='file number to plot', default=0)
    parser.add_argument("--fil_band", type=str, help='folder name for data', default='full_bdw_no_nf')
    parser.add_argument("--version", type=str, help='d0 or d1', default='d0')
    parser.add_argument("--folder", type=str, help='if d1, folder within d1', default=' ')
    args = parser.parse_args()

    plot_waveform(args.file_num, args.fil_band, args.version, args.folder)