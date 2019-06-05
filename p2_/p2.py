from functions import *
from info_file import info_file


def p2(start, end, date, date_time, filter_band, nhdr, fsps, r, pmt_hv, gain, offset, trig_delay, amp,
       band, nfilter):
    gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
    save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (date, filter_band))
    data_path = Path(save_path / 'd1')
    dest_path = Path(save_path / 'd2')
    filt_path1 = Path(dest_path / 'filter1')
    filt_path2 = Path(dest_path / 'filter2')
    filt_path4 = Path(dest_path / 'filter4')
    filt_path8 = Path(dest_path / 'filter8')

    x1_array = np.array([])
    j_array = np.array([])
    error_array = np.array([])

    for i in range(start, end + 1):
        file_name = str(data_path / 'd1_shifted' / 'D1--waveforms--%05d.txt') % i
        save_name1 = str(filt_path1 / 'D2--waveforms--%05d.txt') % i
        save_name2 = str(filt_path2 / 'D2--waveforms--%05d.txt') % i
        save_name4 = str(filt_path4 / 'D2--waveforms--%05d.txt') % i
        save_name8 = str(filt_path8 / 'D2--waveforms--%05d.txt') % i

        if os.path.isfile(file_name):
            if os.path.isfile(save_name1) and os.path.isfile(save_name2) and os.path.isfile(save_name4) and \
                    os.path.isfile(save_name8):
                print('File #%05d has been filtered' % i)
            else:
                print('Filtering file #%05d' % i)
                try:
                    t, v, hdr = rw(file_name, nhdr)
                    rt1090 = rise_time_1090(t, v)
                    print('rise time is', rt1090, 's')
                    for k in range(5, 8000):
                        j = k * 2e-10
                        v_new = lowpass_filter(v, j, fsps)
                        x1 = rise_time_1090(t, v_new)
                        x1_array = np.append(x1_array, x1)
                        j_array = np.append(j_array, j)
                        diff_val = x1 - 8 * rt1090
                        if diff_val >= 0:
                            break
                        if k == 7999:
                            error_array = np.append(error_array, i)
                except Exception:
                    error_array = np.append(error_array, i)

                val = 0
                for j in range(len(error_array)):
                    if error_array[j] == i:
                        val = 1
                        break
                if val == 0:
                    tau_2 = j_array[np.argmin(np.abs(x1_array - 2 * rt1090))]
                    tau_4 = j_array[np.argmin(np.abs(x1_array - 4 * rt1090))]
                    tau_8 = j_array[np.argmin(np.abs(x1_array - 8 * rt1090))]
                    print('tau of 2x rise time is', tau_2)
                    print('tau of 4x rise time is', tau_4)
                    print('tau of 8x rise time is', tau_8)
                    v2 = lowpass_filter(v, tau_2, fsps)
                    v4 = lowpass_filter(v, tau_4, fsps)
                    v8 = lowpass_filter(v, tau_8, fsps)
                    ww(t, v, save_name1, hdr)
                    ww(t, v2, save_name2, hdr)
                    ww(t, v4, save_name4, hdr)
                    ww(t, v8, save_name8, hdr)

    average_waveform(start, end, filt_path1, dest_path, nhdr, 'avg_waveform1')
    average_waveform(start, end, filt_path2, dest_path, nhdr, 'avg_waveform2')
    average_waveform(start, end, filt_path4, dest_path, nhdr, 'avg_waveform4')
    average_waveform(start, end, filt_path8, dest_path, nhdr, 'avg_waveform8')

    avg_path = Path(dest_path / 'plots')
    t1, v1, hdr1 = rw(avg_path / 'avg_waveform1', nhdr)
    t2, v2, hdr2 = rw(avg_path / 'avg_waveform2', nhdr)
    t3, v3, hdr3 = rw(avg_path / 'avg_waveform3', nhdr)
    t4, v4, hdr4 = rw(avg_path / 'avg_waveform4', nhdr)

    rt1 = rise_time_1090(t1, v1)
    rt2 = rise_time_1090(t2, v2)
    rt3 = rise_time_1090(t3, v3)
    rt4 = rise_time_1090(t4, v4)

    print('Average rise time is', rt1)
    print('Average 2x rise time is', rt2)
    print('Average 4x rise time is', rt3)
    print('Average 8x rise time is', rt4)

    info_file(date_time, data_path, dest_path, pmt_hv, gain, offset, trig_delay, amp, fsps, band, nfilter, r)

    print(error_array)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="p2", description="Creating D2")
    parser.add_argument("--start", type=int, help='file number to begin at (default=0)', default=0)
    parser.add_argument("--end", type=int, help='file number to end at (default=99999)', default=99999)
    parser.add_argument("--date", type=int, help='date of data acquisition (YEARMMDD)')
    parser.add_argument("--date_time", type=str, help='date & time of d1 processing')
    parser.add_argument("--fil_band", type=str, help='folder name for data')
    parser.add_argument("--nhdr", type=int, help='number of header lines to skip in raw file (default=5)', default=5)
    parser.add_argument("--fsps", type=float, help='samples per second (Hz) (suggested=20000000000.)')
    parser.add_argument("--r", type=int, help='resistance in ohms (suggested=50)')
    parser.add_argument("--pmt_hv", type=int, help='voltage of PMT (V) (suggested=1800)')
    parser.add_argument("--gain", type=int, help='gain of PMT (suggested=1e7)')
    parser.add_argument("--offset", type=int, help='offset of pulse generator (suggested=0)')
    parser.add_argument("--trig_delay", type=float, help='delay of pulse generator trigger (ns) (suggested=9.)')
    parser.add_argument("--amp", type=float, help='amplitude of pulse generator (V) (suggested=3.5)')
    parser.add_argument("--band", type=str, help='bandwidth of oscilloscope (Hz)')
    parser.add_argument("--nfilter", type=float, help='noise filter on oscilloscope (bits)')
    parser.add_argument("--info_file", type=str, help='path to d1 info file')
    args = parser.parse_args()

    if not args.info_file:
        if not (args.date or args.date_time or args.fil_band or args.fsps or args.baseline or args.r or args.pmt_hv or
                args.gain or args.offset or args.trig_delay or args.amp or args.band or args.nfilter):
            print('Error: Must provide an info file or all other arguments')
        else:
            p2(args.start, args.end, args.date, args.date_time, args.fil_band, args.nhdr, args.fsps, args.r,
               args.pmt_hv, args.gain, args.offset, args.trig_delay, args.amp, args.band, args.nfilter)
    else:
        myfile = open(args.info_file, 'r')
        csv_reader = csv.reader(myfile)
        info_array = np.array([])
        path_array = np.array([])
        for row in csv_reader:
            info_array = np.append(info_array, row[1])
        i_date_time = info_array[1]
        i_path = info_array[3]
        i_pmt_hv = int(info_array[4])
        i_gain = int(float(info_array[5]))
        i_offset = int(info_array[6])
        i_trig_delay = float(info_array[7])
        i_amp = float(info_array[8])
        i_fsps = float(info_array[9])
        i_band = info_array[10]
        i_nfilter = float(info_array[11])
        i_r = int(info_array[12])
        a, b, c, d, e, fol, f, i_fil_band, g = i_path.split('/')
        i_date, watch, spe = fol.split('_')
        i_date = int(i_date)

        p2(args.start, args.end, i_date, i_date_time, i_fil_band, args.nhdr, i_fsps, i_r, i_pmt_hv, i_gain, i_offset,
           i_trig_delay, i_amp, i_band, i_nfilter)

        myfile.close()

# 4991