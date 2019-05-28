from functions import *


# Calculates the average waveform of an spe
def average_waveform(start, end, dest_path, nhdr):
    data_file = Path(dest_path / 'd1_shifted')
    save_file = Path(dest_path / 'plots')
    tsum = 0
    vsum = 0
    n = 0
    for i in range(start, end + 1):
        file_name = 'D1--waveforms--%05d.txt' % i
        if os.path.isfile(data_file / file_name):
            print('File #', i)
            t, v, hdr = rw(data_file / file_name, nhdr)
            v = v / min(v)
            idx = np.where(t == 0)
            idx = int(idx[0])
            t = np.roll(t, -idx)
            v = np.roll(v, -idx)
            idx2 = np.where(t == min(t))
            idx2 = int(idx2[0])
            idx3 = np.where(t == max(t))
            idx3 = int(idx3[0])
            if idx2 <= 3430:
                t = np.concatenate((t[:idx3], t[3431:]))
                v = np.concatenate((v[:idx3], v[3431:]))
                t = np.roll(t, -idx3)
                v = np.roll(v, -idx3)
                if len(t) >= 3920:
                    t = t[:3920]
                    v = v[:3920]
                    tsum += t
                    vsum += v
                    n += 1
    t_avg = tsum / n
    v_avg = vsum / n

    plt.plot(t_avg, v_avg)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Voltage')
    plt.title('Average Waveform')
    plt.savefig(save_file / 'avg_waveform.png', dpi=360)
    plt.show()

    file_name = dest_path / 'hist_data' / 'avg_waveform.txt'
    hdr = 'Average Waveform\n\n\n\nTime,Ampl'
    ww(t_avg, v_avg, file_name, hdr)


if __name__ == '__main__':
    path_d1 = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/full_bdw_no_nf/d1')
    import argparse
    parser = argparse.ArgumentParser(prog='average_waveform', description='creating plot of average waveform shape')
    parser.add_argument('--start', type=int, help='file number to begin at', default=0)
    parser.add_argument('--end', type=int, help='file number to end at', default=99999)
    parser.add_argument('--dest_path', type=str, help='path to d1 folder', default=path_d1)
    parser.add_argument('--nhdr', type=int, help='number of lines to ignore for header', default=5)
    args = parser.parse_args()

    average_waveform(args.start, args.end, args.dest_path, args.nhdr)
