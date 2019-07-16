from functions import *

gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (20190513, 'full_bdw_no_nf'))
dest_path = Path(save_path / 'd2')
filt_path = Path(dest_path / 'rt_1')
double_path = Path(dest_path / 'double_spe')
single_path = Path(dest_path / 'single_spe')

double_file_array = np.array([])

min_amp_1 = 1
min_amp_2 = 1
min_amp_4 = 1
min_amp_8 = 1

t, v, hdr = rw(dest_path / 'hist_data' / 'avg_waveform_double_rt1_40_ns.txt', 5)
plt.plot(t, v)
plt.show()

print('Checking existing double spe files...')
for filename in os.listdir(double_path / 'rt_1' / '40_ns'):
    print(filename, 'is a file')
    files_added = filename[15:27]
    double_file_array = np.append(double_file_array, files_added)

for item in double_file_array:
    file_name1 = str(dest_path / 'double_spe' / 'rt_1' / '40_ns' / 'D2--waveforms--%s.txt') % item
    file_name2 = str(dest_path / 'double_spe' / 'rt_2' / '40_ns' / 'D2--waveforms--%s.txt') % item
    file_name4 = str(dest_path / 'double_spe' / 'rt_4' / '40_ns' / 'D2--waveforms--%s.txt') % item
    file_name8 = str(dest_path / 'double_spe' / 'rt_8' / '40_ns' / 'D2--waveforms--%s.txt') % item

    if os.path.isfile(file_name1) and os.path.isfile(file_name2) and os.path.isfile(file_name4) and \
            os.path.isfile(file_name8):
        print("Calculating file #%s" % item)

        t1, v1, hdr1 = rw(file_name1, 5)            # Unfiltered waveform file is read
        peak_amts = np.array([])
        idx1 = np.inf
        idx2 = np.inf
        try:
            v_flip = -1 * v1
            peaks, _ = signal.find_peaks(v_flip, max(v_flip) / 20)
            for thing in peaks:
                peak_amts = np.append(peak_amts, v_flip[thing])
            true_max = v1[peaks[np.where(peak_amts == max(peak_amts))]][0]
            if len(np.where(peak_amts == max(peak_amts))[0]) == 1:
                peak_amts[np.where(peak_amts == max(peak_amts))] = 0
            else:
                peak_amts[np.where(peak_amts == max(peak_amts))[0][0]] = 0
            sec_max = v1[peaks[np.where(peak_amts == max(peak_amts))]][0]
            v_sum = 0
            for i in range(251):
                v_sum += v1[i]
            avg = v_sum / 250
            if sec_max >= min_amp_1 or len(peaks) == 1:
                tvals_1 = np.linspace(t1[0], t1[len(t1) - 1], 1000)
                vvals_1 = np.interp(tvals_1, t1, v1)
                idx_max_peak = np.argmin(np.abs(vvals_1 - true_max)).item()
                t_rising_1 = tvals_1[:idx_max_peak].item()
                v_rising_1 = vvals_1[:idx_max_peak].item()
                val10_1 = .1 * (true_max - avg)             # Calculates 10% max
                val90_1 = 9 * val10_1                       # Calculates 90% max
                time10_1 = t_rising_1[np.argmin(np.abs(v_rising_1 - val10_1))]    # Finds time of point of 10% max
                time90_1 = t_rising_1[np.argmin(np.abs(v_rising_1 - val90_1))]    # Finds time of point of 90% max
                rise_time1090_1 = float(format(time90_1 - time10_1, '.2e'))         # Calculates 10-90 rise time
                x1 = 1
            else:
                if np.where(v1 == true_max) < np.where(v1 == sec_max):
                    tvals_1 = np.linspace(t1[0], t1[np.where(v1 == true_max)], 500)
                    vvals_1 = np.interp(tvals_1, t1, v1)
                    idx_max_peak = np.argmin(np.abs(vvals_1 - true_max)).item()
                    t_rising_1 = tvals_1[:idx_max_peak].item()
                    v_rising_1 = vvals_1[:idx_max_peak].item()
                    val10_1 = .1 * (true_max - avg)             # Calculates 10% max
                    val90_1 = 9 * val10_1                       # Calculates 90% max
                    time10_1 = t_rising_1[np.argmin(np.abs(v_rising_1 - val10_1))]  # Finds time of point of 10% max
                    time90_1 = t_rising_1[np.argmin(np.abs(v_rising_1 - val90_1))]  # Finds time of point of 90% max
                    x1 = 1
                elif np.where(v1 == sec_max) < np.where(v1 == true_max):
                    tvals_1 = np.linspace(t1[0], t1[np.where(v1 == sec_max)], 500)
                    vvals_1 = np.interp(tvals_1, t1, v1)
                    idx_max_peak = np.argmin(np.abs(vvals_1 - sec_max)).item()
                    t_rising_1 = tvals_1[:idx_max_peak].item()
                    v_rising_1 = vvals_1[:idx_max_peak].item()
                    val10_1 = .1 * (true_max - avg)             # Calculates 10% max
                    val90_1 = 9 * val10_1                       # Calculates 90% max
                    time10_1 = t_rising_1[np.argmin(np.abs(v_rising_1 - val10_1))]  # Finds time of point of 10% max
                    time90_1 = t_rising_1[np.argmin(np.abs(v_rising_1 - val90_1))]  # Finds time of point of 90% max
                    x1 = 1
                else:
                    time10_1 = 0
                    time90_1 = -1
                rise_time1090_1 = float(format(time90_1 - time10_1, '.2e'))     # Calculates 10-90 rise time
        except Exception:
            rise_time1090_1 = -1

        t2, v2, hdr2 = rw(file_name2, 5)        # 2x shaped waveform file is read
        peak_amts = np.array([])
        idx1 = np.inf
        idx2 = np.inf
        try:
            v_flip = -1 * v2
            peaks, _ = signal.find_peaks(v_flip, max(v_flip) / 20)
            for thing in peaks:
                peak_amts = np.append(peak_amts, v_flip[thing])
            true_max = v2[peaks[np.where(peak_amts == max(peak_amts))]][0]
            if len(np.where(peak_amts == max(peak_amts))[0]) == 1:
                peak_amts[np.where(peak_amts == max(peak_amts))] = 0
            else:
                peak_amts[np.where(peak_amts == max(peak_amts))[0][0]] = 0
            sec_max = v2[peaks[np.where(peak_amts == max(peak_amts))]][0]
            v_sum = 0
            for i in range(251):
                v_sum += v2[i]
            avg = v_sum / 250
            if sec_max >= min_amp_2 or len(peaks) == 1:
                tvals_2 = np.linspace(t2[0], t2[len(t2) - 1], 1000)
                vvals_2 = np.interp(tvals_2, t2, v2)
                idx_max_peak = np.argmin(np.abs(vvals_2 - true_max)).item()
                t_rising_2 = tvals_2[:idx_max_peak].item()
                v_rising_2 = vvals_2[:idx_max_peak].item()
                val10_2 = .1 * (true_max - avg)  # Calculates 10% max
                val90_2 = 9 * val10_2  # Calculates 90% max
                time10_2 = t_rising_2[np.argmin(np.abs(v_rising_2 - val10_2))]  # Finds time of point of 10% max
                time90_2 = t_rising_2[np.argmin(np.abs(v_rising_2 - val90_2))]  # Finds time of point of 90% max
                rise_time1090_2 = float(format(time90_2 - time10_2, '.2e'))  # Calculates 10-90 rise time
                x2 = 1
            else:
                if np.where(v2 == true_max) < np.where(v2 == sec_max):
                    tvals_2 = np.linspace(t2[0], t2[np.where(v2 == true_max)], 500)
                    vvals_2 = np.interp(tvals_2, t2, v2)
                    idx_max_peak = np.argmin(np.abs(vvals_2 - true_max)).item()
                    t_rising_2 = tvals_2[:idx_max_peak].item()
                    v_rising_2 = vvals_2[:idx_max_peak].item()
                    val10_2 = .1 * (true_max - avg)  # Calculates 10% max
                    val90_2 = 9 * val10_2  # Calculates 90% max
                    time10_2 = t_rising_2[np.argmin(np.abs(v_rising_2 - val10_2))]  # Finds time of point of 10% max
                    time90_2 = t_rising_2[np.argmin(np.abs(v_rising_2 - val90_2))]  # Finds time of point of 90% max
                    x2 = 1
                elif np.where(v2 == sec_max) < np.where(v2 == true_max):
                    tvals_2 = np.linspace(t2[0], t2[np.where(v2 == sec_max)], 500)
                    vvals_2 = np.interp(tvals_2, t2, v2)
                    idx_max_peak = np.argmin(np.abs(vvals_2 - sec_max)).item()
                    t_rising_2 = tvals_2[:idx_max_peak].item()
                    v_rising_2 = vvals_2[:idx_max_peak].item()
                    val10_2 = .1 * (true_max - avg)  # Calculates 10% max
                    val90_2 = 9 * val10_2  # Calculates 90% max
                    time10_2 = t_rising_2[np.argmin(np.abs(v_rising_2 - val10_2))]  # Finds time of point of 10% max
                    time90_2 = t_rising_2[np.argmin(np.abs(v_rising_2 - val90_2))]  # Finds time of point of 90% max
                    x2 = 1
                else:
                    time10_2 = 0
                    time90_2 = -1
                rise_time1090_2 = float(format(time90_2 - time10_2, '.2e'))  # Calculates 10-90 rise time
        except Exception:
            rise_time1090_2 = -1

        t4, v4, hdr4 = rw(file_name4, 5)        # 4x shaped waveform file is read
        peak_amts = np.array([])
        idx1 = np.inf
        idx2 = np.inf
        try:
            v_flip = -1 * v4
            peaks, _ = signal.find_peaks(v_flip, max(v_flip) / 20)
            for thing in peaks:
                peak_amts = np.append(peak_amts, v_flip[thing])
            true_max = v4[peaks[np.where(peak_amts == max(peak_amts))]][0]
            if len(np.where(peak_amts == max(peak_amts))[0]) == 1:
                peak_amts[np.where(peak_amts == max(peak_amts))] = 0
            else:
                peak_amts[np.where(peak_amts == max(peak_amts))[0][0]] = 0
            sec_max = v4[peaks[np.where(peak_amts == max(peak_amts))]][0]
            v_sum = 0
            for i in range(251):
                v_sum += v4[i]
            avg = v_sum / 250
            if sec_max >= min_amp_4 or len(peaks) == 1:
                tvals_4 = np.linspace(t4[0], t4[len(t4) - 1], 1000)
                vvals_4 = np.interp(tvals_4, t4, v4)
                idx_max_peak = np.argmin(np.abs(vvals_4 - true_max)).item()
                t_rising_4 = tvals_4[:idx_max_peak].item()
                v_rising_4 = vvals_4[:idx_max_peak].item()
                val10_4 = .1 * (true_max - avg)  # Calculates 10% max
                val90_4 = 9 * val10_4  # Calculates 90% max
                time10_4 = t_rising_4[np.argmin(np.abs(v_rising_4 - val10_4))]  # Finds time of point of 10% max
                time90_4 = t_rising_4[np.argmin(np.abs(v_rising_4 - val90_4))]  # Finds time of point of 90% max
                rise_time1090_4 = float(format(time90_4 - time10_4, '.2e'))  # Calculates 10-90 rise time
                x4 = 1
            else:
                if np.where(v4 == true_max) < np.where(v4 == sec_max):
                    tvals_4 = np.linspace(t4[0], t4[np.where(v4 == true_max)], 500)
                    vvals_4 = np.interp(tvals_4, t4, v4)
                    idx_max_peak = np.argmin(np.abs(vvals_4 - true_max)).item()
                    t_rising_4 = tvals_4[:idx_max_peak].item()
                    v_rising_4 = vvals_4[:idx_max_peak].item()
                    val10_4 = .1 * (true_max - avg)  # Calculates 10% max
                    val90_4 = 9 * val10_4  # Calculates 90% max
                    time10_4 = t_rising_4[np.argmin(np.abs(v_rising_4 - val10_4))]  # Finds time of point of 10% max
                    time90_4 = t_rising_4[np.argmin(np.abs(v_rising_4 - val90_4))]  # Finds time of point of 90% max
                    x4 = 1
                elif np.where(v4 == sec_max) < np.where(v4 == true_max):
                    tvals_4 = np.linspace(t4[0], t4[np.where(v4 == sec_max)], 500)
                    vvals_4 = np.interp(tvals_4, t4, v4)
                    idx_max_peak = np.argmin(np.abs(vvals_4 - sec_max)).item()
                    t_rising_4 = tvals_4[:idx_max_peak].item()
                    v_rising_4 = vvals_4[:idx_max_peak].item()
                    val10_4 = .1 * (true_max - avg)  # Calculates 10% max
                    val90_4 = 9 * val10_4  # Calculates 90% max
                    time10_4 = t_rising_4[np.argmin(np.abs(v_rising_4 - val10_4))]  # Finds time of point of 10% max
                    time90_4 = t_rising_4[np.argmin(np.abs(v_rising_4 - val90_4))]  # Finds time of point of 90% max
                    x4 = 1
                else:
                    time10_4 = 0
                    time90_4 = -1
                rise_time1090_4 = float(format(time90_4 - time10_4, '.2e'))  # Calculates 10-90 rise time
        except Exception:
            rise_time1090_4 = -1

        t8, v8, hdr8 = rw(file_name8, 5)        # 8x shaped waveform file is read
        peak_amts = np.array([])
        idx1 = np.inf
        idx2 = np.inf
        try:
            v_flip = -1 * v8
            peaks, _ = signal.find_peaks(v_flip, max(v_flip) / 20)
            for thing in peaks:
                peak_amts = np.append(peak_amts, v_flip[thing])
            true_max = v8[peaks[np.where(peak_amts == max(peak_amts))]][0]
            if len(np.where(peak_amts == max(peak_amts))[0]) == 1:
                peak_amts[np.where(peak_amts == max(peak_amts))] = 0
            else:
                peak_amts[np.where(peak_amts == max(peak_amts))[0][0]] = 0
            sec_max = v8[peaks[np.where(peak_amts == max(peak_amts))]][0]
            v_sum = 0
            for i in range(251):
                v_sum += v8[i]
            avg = v_sum / 250
            if sec_max >= min_amp_8 or len(peaks) == 1:
                tvals_8 = np.linspace(t8[0], t8[len(t8) - 1], 1000)
                vvals_8 = np.interp(tvals_8, t8, v8)
                idx_max_peak = np.argmin(np.abs(vvals_8 - true_max)).item()
                t_rising_8 = tvals_8[:idx_max_peak].item()
                v_rising_8 = vvals_8[:idx_max_peak].item()
                val10_8 = .1 * (true_max - avg)  # Calculates 10% max
                val90_8 = 9 * val10_8  # Calculates 90% max
                time10_8 = t_rising_8[np.argmin(np.abs(v_rising_8 - val10_8))]  # Finds time of point of 10% max
                time90_8 = t_rising_8[np.argmin(np.abs(v_rising_8 - val90_8))]  # Finds time of point of 90% max
                rise_time1090_8 = float(format(time90_8 - time10_8, '.2e'))  # Calculates 10-90 rise time
                x8 = 1
            else:
                if np.where(v8 == true_max) < np.where(v8 == sec_max):
                    tvals_8 = np.linspace(t8[0], t8[np.where(v8 == true_max)], 500)
                    vvals_8 = np.interp(tvals_8, t8, v8)
                    idx_max_peak = np.argmin(np.abs(vvals_8 - true_max)).item()
                    t_rising_8 = tvals_8[:idx_max_peak].item()
                    v_rising_8 = vvals_8[:idx_max_peak].item()
                    val10_8 = .1 * (true_max - avg)  # Calculates 10% max
                    val90_8 = 9 * val10_8  # Calculates 90% max
                    time10_8 = t_rising_8[np.argmin(np.abs(v_rising_8 - val10_8))]  # Finds time of point of 10% max
                    time90_8 = t_rising_8[np.argmin(np.abs(v_rising_8 - val90_8))]  # Finds time of point of 90% max
                    x8 = 1
                elif np.where(v8 == sec_max) < np.where(v8 == true_max):
                    tvals_8 = np.linspace(t8[0], t8[np.where(v8 == sec_max)], 500)
                    vvals_8 = np.interp(tvals_8, t8, v8)
                    idx_max_peak = np.argmin(np.abs(vvals_8 - sec_max)).item()
                    t_rising_8 = tvals_8[:idx_max_peak].item()
                    v_rising_8 = vvals_8[:idx_max_peak].item()
                    val10_8 = .1 * (true_max - avg)  # Calculates 10% max
                    val90_8 = 9 * val10_8  # Calculates 90% max
                    time10_8 = t_rising_8[np.argmin(np.abs(v_rising_8 - val10_8))]  # Finds time of point of 10% max
                    time90_8 = t_rising_8[np.argmin(np.abs(v_rising_8 - val90_8))]  # Finds time of point of 90% max
                    x8 = 1
                else:
                    time10_8 = 0
                    time90_8 = -1
                rise_time1090_8 = float(format(time90_8 - time10_8, '.2e'))  # Calculates 10-90 rise time
        except Exception:
            rise_time1090_8 = -1

        if rise_time1090_1 <= 0 and x1 == 1:
            print(item, 'rt 1', rise_time1090_1)
            plt.plot(t1, v1)
            plt.plot(time10_1, val10_1, 'x')
            plt.plot(time90_1, val90_1, 'x')
            plt.show()
        elif rise_time1090_1 <= 0:
            print(item, 'rt 1')
            plt.plot(t1, v1)
            plt.show()

        if rise_time1090_2 <= 0 and x2 == 1:
            print(item, 'rt 2', rise_time1090_2)
            plt.plot(t2, v2)
            plt.plot(time10_2, val10_2, 'x')
            plt.plot(time90_2, val90_2, 'x')
            plt.show()
        elif rise_time1090_2 <= 0:
            print(item, 'rt 2')
            plt.plot(t2, v2)
            plt.show()

        if rise_time1090_4 <= 0 and x4 == 1:
            print(item, 'rt 4', rise_time1090_4)
            plt.plot(t4, v4)
            plt.plot(time10_4, val10_4, 'x')
            plt.plot(time90_4, val90_4, 'x')
            plt.show()
        elif rise_time1090_4 <= 0:
            print(item, 'rt 4')
            plt.plot(t4, v4)
            plt.show()

        if rise_time1090_8 <= 0 and x8 == 1:
            print(item, 'rt 8', rise_time1090_8)
            plt.plot(t8, v8)
            plt.plot(time10_8, val10_8, 'x')
            plt.plot(time90_8, val90_8, 'x')
            plt.show()
        elif rise_time1090_8 <= 0:
            print(item, 'rt 8')
            plt.plot(t8, v8)
            plt.show()
