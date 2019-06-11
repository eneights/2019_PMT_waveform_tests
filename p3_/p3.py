from functions import *
from info_file import info_file


# Downsamples and digitizes spe waveforms
def p3(start, end, date, date_time, filter_band, nhdr, fsps, r, pmt_hv, gain, offset, trig_delay, amp, band, nfilter,
       fsps_new, noise):
    gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
    save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (date, filter_band))
    data_path = Path(save_path / 'd2')
    dest_path = Path(save_path / 'd3')
    filt_path_1 = Path(dest_path / 'rt_1')
    filt_path_2 = Path(dest_path / 'rt_2')
    filt_path_4 = Path(dest_path / 'rt_4')
    filt_path_8 = Path(dest_path / 'rt_8')

    # Copies waveforms with 1x, 2x, 4x, and 8x initial rise times to d3 folder
    for i in range(start, end + 1):
        file_name1 = str(data_path / 'filter1' / 'D2--waveforms--%05d.txt') % i
        file_name2 = str(data_path / 'filter2' / 'D2--waveforms--%05d.txt') % i
        file_name4 = str(data_path / 'filter2_2' / 'D2--waveforms--%05d.txt') % i
        file_name8 = str(data_path / 'filter2_2_2' / 'D2--waveforms--%05d.txt') % i
        save_name1 = str(filt_path_1 / 'raw' / 'D3--waveforms--%05d.txt') % i
        save_name2 = str(filt_path_2 / 'raw' / 'D3--waveforms--%05d.txt') % i
        save_name4 = str(filt_path_4 / 'raw' / 'D3--waveforms--%05d.txt') % i
        save_name8 = str(filt_path_8 / 'raw' / 'D3--waveforms--%05d.txt') % i

        if os.path.isfile(file_name1):
            if os.path.isfile(save_name1):
                print('File #%05d in rt_1 folder' % i)
            else:
                t, v, hdr = rw(file_name1, nhdr)
                ww(t, v, save_name1, hdr)
                print('File #%05d in rt_1 folder' % i)

        if os.path.isfile(file_name2):
            if os.path.isfile(save_name2):
                print('File #%05d in rt_2 folder' % i)
            else:
                t, v, hdr = rw(file_name2, nhdr)
                ww(t, v, save_name2, hdr)
                print('File #%05d in rt_2 folder' % i)

        if os.path.isfile(file_name4):
            if os.path.isfile(save_name4):
                print('File #%05d in rt_4 folder' % i)
            else:
                t, v, hdr = rw(file_name4, nhdr)
                ww(t, v, save_name4, hdr)
                print('File #%05d in rt_4 folder' % i)

        if os.path.isfile(file_name8):
            if os.path.isfile(save_name8):
                print('File #%05d in rt_8 folder' % i)
            else:
                t, v, hdr = rw(file_name8, nhdr)
                ww(t, v, save_name8, hdr)
                print('File #%05d in rt_8 folder' % i)

    # Downsamples waveforms using given fsps
    for i in range(start, end + 1):
        file_name1 = str(filt_path_1 / 'raw' / 'D3--waveforms--%05d.txt') % i
        file_name2 = str(filt_path_2 / 'raw' / 'D3--waveforms--%05d.txt') % i
        file_name4 = str(filt_path_4 / 'raw' / 'D3--waveforms--%05d.txt') % i
        file_name8 = str(filt_path_8 / 'raw' / 'D3--waveforms--%05d.txt') % i
        save_name1 = str(filt_path_1 / 'downsampled' / 'D3--waveforms--%05d.txt') % i
        save_name2 = str(filt_path_2 / 'downsampled' / 'D3--waveforms--%05d.txt') % i
        save_name4 = str(filt_path_4 / 'downsampled' / 'D3--waveforms--%05d.txt') % i
        save_name8 = str(filt_path_8 / 'downsampled' / 'D3--waveforms--%05d.txt') % i

        if os.path.isfile(save_name1) and os.path.isfile(save_name2) and os.path.isfile(save_name4) and \
                os.path.isfile(save_name8):
            print('File #%05d downsampled' % i)
        else:
            if os.path.isfile(file_name1) and os.path.isfile(file_name2) and os.path.isfile(file_name4) and \
                    os.path.isfile(file_name8):
                print('Downsampling file #%05d' % i)
                if not os.path.isfile(save_name1):
                    t, v, hdr = rw(file_name1, nhdr)
                    t_ds, v_ds = downsample(t, v, fsps, fsps_new)
                    ww(t_ds, v_ds, save_name1, hdr)
                if not os.path.isfile(save_name2):
                    t, v, hdr = rw(file_name2, nhdr)
                    t_ds, v_ds = downsample(t, v, fsps, fsps_new)
                    ww(t_ds, v_ds, save_name2, hdr)
                if not os.path.isfile(save_name4):
                    t, v, hdr = rw(file_name4, nhdr)
                    t_ds, v_ds = downsample(t, v, fsps, fsps_new)
                    ww(t_ds, v_ds, save_name4, hdr)
                if not os.path.isfile(save_name8):
                    t, v, hdr = rw(file_name8, nhdr)
                    t_ds, v_ds = downsample(t, v, fsps, fsps_new)
                    ww(t_ds, v_ds, save_name8, hdr)

    # Digitizes waveforms using given noise
    for i in range(start, end + 1):
        file_name1 = str(filt_path_1 / 'downsampled' / 'D3--waveforms--%05d.txt') % i
        file_name2 = str(filt_path_2 / 'downsampled' / 'D3--waveforms--%05d.txt') % i
        file_name4 = str(filt_path_4 / 'downsampled' / 'D3--waveforms--%05d.txt') % i
        file_name8 = str(filt_path_8 / 'downsampled' / 'D3--waveforms--%05d.txt') % i
        save_name1 = str(filt_path_1 / 'digitized' / 'D3--waveforms--%05d.txt') % i
        save_name2 = str(filt_path_2 / 'digitized' / 'D3--waveforms--%05d.txt') % i
        save_name4 = str(filt_path_4 / 'digitized' / 'D3--waveforms--%05d.txt') % i
        save_name8 = str(filt_path_8 / 'digitized' / 'D3--waveforms--%05d.txt') % i

        if os.path.isfile(save_name1) and os.path.isfile(save_name2) and os.path.isfile(save_name4) and \
                os.path.isfile(save_name8):
            print('File #%05d digitized' % i)
        else:
            if os.path.isfile(file_name1) and os.path.isfile(file_name2) and os.path.isfile(file_name4) and \
                    os.path.isfile(file_name8):
                print('Digitizing file #%05d' % i)
                if not os.path.isfile(save_name1):
                    t, v, hdr = rw(file_name1, nhdr)
                    v_dig = digitize(v, noise)
                    ww(t, v_dig, save_name1, hdr)
                if not os.path.isfile(save_name2):
                    t, v, hdr = rw(file_name2, nhdr)
                    v_dig = digitize(v, noise)
                    ww(t, v_dig, save_name2, hdr)
                if not os.path.isfile(save_name4):
                    t, v, hdr = rw(file_name4, nhdr)
                    v_dig = digitize(v, noise)
                    ww(t, v_dig, save_name4, hdr)
                if not os.path.isfile(save_name8):
                    t, v, hdr = rw(file_name8, nhdr)
                    v_dig = digitize(v, noise)
                    ww(t, v_dig, save_name8, hdr)

    # Plots average waveforms for 1x, 2x, 4x, and 8x initial rise times after downsampling and digitizing
    print('Calculating rt_1 average waveforms...')
    average_waveform(start, end, filt_path_1 / 'downsampled', dest_path, nhdr, 'avg_waveform_1_ds')
    average_waveform(start, end, filt_path_1 / 'digitized', dest_path, nhdr, 'avg_waveform_1_dig')

    print('Calculating rt_2 average waveforms...')
    average_waveform(start, end, filt_path_2 / 'downsampled', dest_path, nhdr, 'avg_waveform_2_ds')
    average_waveform(start, end, filt_path_2 / 'digitized', dest_path, nhdr, 'avg_waveform_2_dig')

    print('Calculating rt_4 average waveforms...')
    average_waveform(start, end, filt_path_4 / 'downsampled', dest_path, nhdr, 'avg_waveform_4_ds')
    average_waveform(start, end, filt_path_4 / 'digitized', dest_path, nhdr, 'avg_waveform_4_dig')

    print('Calculating rt_8 average waveforms...')
    average_waveform(start, end, filt_path_8 / 'downsampled', dest_path, nhdr, 'avg_waveform_8_ds')
    average_waveform(start, end, filt_path_8 / 'digitized', dest_path, nhdr, 'avg_waveform_8_dig')

    # Writes info file
    info_file(date_time, source_path, dest_path, pmt_hv, gain, offset, trig_delay, amp, fsps, band, nfilter, r)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="p3", description="Creating D3")
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
    parser.add_argument("--fsps_new", type=float, help='new samples per second (Hz) (suggested=500000000.)')
    parser.add_argument("--noise", type=float, help='noise to add (bits) (suggested=3.30)')
    parser.add_argument("--info_file", type=str, help='path to d2 info file')
    args = parser.parse_args()

    if not args.info_file:
        if not (args.date or args.date_time or args.fil_band or args.fsps or args.r or args.pmt_hv or
                args.gain or args.offset or args.trig_delay or args.amp or args.band or args.nfilter, args.fsps_new,
                args.noise):
            print('Error: Must provide an info file or all other arguments')
        else:
            p3(args.start, args.end, args.date, args.date_time, args.fil_band, args.nhdr, args.fsps, args.r,
               args.pmt_hv, args.gain, args.offset, args.trig_delay, args.amp, args.band, args.nfilter, args.fsps_new,
                args.noise)
    elif not (args.fsps_new or args.noise):
        print('Error: Must provide new fsps and noise level')
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

        p3(args.start, args.end, i_date, i_date_time, i_fil_band, args.nhdr, i_fsps, i_r, i_pmt_hv, i_gain, i_offset,
           i_trig_delay, i_amp, i_band, i_nfilter, args.fsps_new, args.noise)

        myfile.close()
