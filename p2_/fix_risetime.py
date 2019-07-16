from functions import *

gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (20190513, 'full_bdw_no_nf'))
dest_path = Path(save_path / 'd2')
filt_path = Path(dest_path / 'rt_1')
double_path = Path(dest_path / 'double_spe')
single_path = Path(dest_path / 'single_spe')

double_file_array = np.array([])

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
        v_sum = 0
        for i in range(251):
            v_sum += v1[i]
        avg = v_sum / 250
        idx = np.inf
        idx1 = np.argmin(np.abs(t1))
        idx2 = np.argmin(np.abs(t1 - 5e-8))
        t_spliced = t1[idx1:idx2 + 1]
        v_spliced = v1[idx1:idx2 + 1]
        idx_min_val = np.where(v_spliced == min(v_spliced))  # Finds index of minimum voltage value in voltage array
        time_min_val = t_spliced[idx_min_val]  # Finds time of point of minimum voltage
        min_time = time_min_val[0]
        tvals = np.linspace(t1[0], t1[len(t1) - 1], 5000)  # Creates array of times over entire timespan
        tvals1 = np.linspace(t1[0], min_time, 5000)  # Creates array of times from beginning to point of min voltage
        vvals1 = np.interp(tvals1, t1, v1)
        vvals1_flip = np.flip(vvals1)  # Flips array, creating array of voltages from point of min voltage to beginning
        difference_value = vvals1_flip - (0.1 * (min(v_spliced) - avg))
        for i in range(0, len(difference_value) - 1):
            if difference_value[i] >= 0:  # of waveform, finds where voltage becomes greater than 10% max
                idx = len(difference_value) - i
                break
        if idx == np.inf:  # If voltage never becomes greater than 10% max, finds where voltage is closest to 10% max
            idx = len(difference_value) - 1 - np.argmin(np.abs(difference_value))
        if idx == 5000:
            rise_time1090_1 = 0
        else:
            t1_1 = tvals[np.argmin(np.abs(tvals - tvals1[idx]))]  # Finds time of beginning of spe
            val10_1 = .1 * (min(v_spliced) - avg)  # Calculates 10% max
            val90_1 = 9 * val10_1  # Calculates 90% max
            tvals2_1 = np.linspace(t1_1, min_time, 5000)
            vvals2_1 = np.interp(tvals2_1, t1, v1)
            time10 = tvals2_1[np.argmin(np.abs(vvals2_1 - val10_1))]  # Finds time of point of 10% max
            time90 = tvals2_1[np.argmin(np.abs(vvals2_1 - val90_1))]  # Finds time of point of 90% max
            rise_time1090_1 = float(format(time90 - time10, '.2e'))  # Calculates 10-90 rise time
            x1 = 1

        t2, v2, hdr2 = rw(file_name2, 5)  # Unfiltered waveform file is read
        v_sum = 0
        for i in range(251):
            v_sum += v2[i]
        avg = v_sum / 250
        idx = np.inf
        idx1 = np.argmin(np.abs(t2))
        idx2 = np.argmin(np.abs(t2 - 6e-8))
        t_spliced = t2[idx1:idx2 + 1]
        v_spliced = v2[idx1:idx2 + 1]
        idx_min_val = np.where(v_spliced == min(v_spliced))  # Finds index of minimum voltage value in voltage array
        time_min_val = t_spliced[idx_min_val]  # Finds time of point of minimum voltage
        min_time = time_min_val[0]
        tvals = np.linspace(t2[0], t2[len(t2) - 1], 5000)  # Creates array of times over entire timespan
        tvals1 = np.linspace(t2[0], min_time, 5000)  # Creates array of times from beginning to point of min voltage
        vvals1 = np.interp(tvals1, t2, v2)
        vvals1_flip = np.flip(vvals1)  # Flips array, creating array of voltages from point of min voltage to beginning
        difference_value = vvals1_flip - (0.1 * (min(v_spliced) - avg))
        for i in range(0,
                       len(difference_value) - 1):  # Starting at point of minimum voltage and going towards beginning
            if difference_value[i] >= 0:  # of waveform, finds where voltage becomes greater than 10% max
                idx = len(difference_value) - i
                break
        if idx == np.inf:  # If voltage never becomes greater than 10% max, finds where voltage is closest to 10% max
            idx = len(difference_value) - 1 - np.argmin(np.abs(difference_value))
        if idx == 5000:
            rise_time1090_2 = 0
        else:
            t1_2 = tvals[np.argmin(np.abs(tvals - tvals1[idx]))]  # Finds time of beginning of spe
            val10_2 = .1 * (min(v_spliced) - avg)  # Calculates 10% max
            val90_2 = 9 * val10_2  # Calculates 90% max
            tvals2_2 = np.linspace(t1_2, min_time, 5000)
            vvals2_2 = np.interp(tvals2_2, t2, v2)
            time10 = tvals2_2[np.argmin(np.abs(vvals2_2 - val10_2))]  # Finds time of point of 10% max
            time90 = tvals2_2[np.argmin(np.abs(vvals2_2 - val90_2))]  # Finds time of point of 90% max
            rise_time1090_2 = float(format(time90 - time10, '.2e'))  # Calculates 10-90 rise time
            x2 = 1

        t4, v4, hdr4 = rw(file_name4, 5)  # Unfiltered waveform file is read
        v_sum = 0
        for i in range(251):
            v_sum += v4[i]
        avg = v_sum / 250
        idx = np.inf
        idx1 = np.argmin(np.abs(t4))
        idx2 = np.argmin(np.abs(t4 - 8e-8))
        t_spliced = t4[idx1:idx2 + 1]
        v_spliced = v4[idx1:idx2 + 1]
        idx_min_val = np.where(v_spliced == min(v_spliced))  # Finds index of minimum voltage value in voltage array
        time_min_val = t_spliced[idx_min_val]  # Finds time of point of minimum voltage
        min_time = time_min_val[0]
        tvals = np.linspace(t4[0], t4[len(t4) - 1], 5000)  # Creates array of times over entire timespan
        tvals1 = np.linspace(t4[0], min_time, 5000)  # Creates array of times from beginning to point of min voltage
        vvals1 = np.interp(tvals1, t4, v4)
        vvals1_flip = np.flip(vvals1)  # Flips array, creating array of voltages from point of min voltage to beginning
        difference_value = vvals1_flip - (0.1 * (min(v_spliced) - avg))
        for i in range(0,
                       len(difference_value) - 1):  # Starting at point of minimum voltage and going towards beginning
            if difference_value[i] >= 0:  # of waveform, finds where voltage becomes greater than 10% max
                idx = len(difference_value) - i
                break
        if idx == np.inf:  # If voltage never becomes greater than 10% max, finds where voltage is closest to 10% max
            idx = len(difference_value) - 1 - np.argmin(np.abs(difference_value))
        if idx == 5000:
            rise_time1090_4 = 0
        else:
            t1_4 = tvals[np.argmin(np.abs(tvals - tvals1[idx]))]  # Finds time of beginning of spe
            val10_4 = .1 * (min(v_spliced) - avg)  # Calculates 10% max
            val90_4 = 9 * val10_4  # Calculates 90% max
            tvals2_4 = np.linspace(t1_4, min_time, 5000)
            vvals2_4 = np.interp(tvals2_4, t4, v4)
            time10 = tvals2_4[np.argmin(np.abs(vvals2_4 - val10_4))]  # Finds time of point of 10% max
            time90 = tvals2_4[np.argmin(np.abs(vvals2_4 - val90_4))]  # Finds time of point of 90% max
            rise_time1090_4 = float(format(time90 - time10, '.2e'))  # Calculates 10-90 rise time
            x4 = 1

        t8, v8, hdr8 = rw(file_name8, 5)  # Unfiltered waveform file is read
        v_sum = 0
        for i in range(251):
            v_sum += v8[i]
        avg = v_sum / 250
        idx = np.inf
        idx1 = np.argmin(np.abs(t8))
        idx2 = np.argmin(np.abs(t8 - 1e-7))
        t_spliced = t8[idx1:idx2 + 1]
        v_spliced = v8[idx1:idx2 + 1]
        idx_min_val = np.where(v_spliced == min(v_spliced))  # Finds index of minimum voltage value in voltage array
        time_min_val = t_spliced[idx_min_val]  # Finds time of point of minimum voltage
        min_time = time_min_val[0]
        tvals = np.linspace(t8[0], t8[len(t8) - 1], 5000)  # Creates array of times over entire timespan
        tvals1 = np.linspace(t8[0], min_time, 5000)  # Creates array of times from beginning to point of min voltage
        vvals1 = np.interp(tvals1, t8, v8)
        vvals1_flip = np.flip(vvals1)  # Flips array, creating array of voltages from point of min voltage to beginning
        difference_value = vvals1_flip - (0.1 * (min(v_spliced) - avg))
        for i in range(0,
                       len(difference_value) - 1):  # Starting at point of minimum voltage and going towards beginning
            if difference_value[i] >= 0:  # of waveform, finds where voltage becomes greater than 10% max
                idx = len(difference_value) - i
                break
        if idx == np.inf:  # If voltage never becomes greater than 10% max, finds where voltage is closest to 10% max
            idx = len(difference_value) - 1 - np.argmin(np.abs(difference_value))
        if idx == 5000:
            rise_time1090_8 = 0
        else:
            t1_8 = tvals[np.argmin(np.abs(tvals - tvals1[idx]))]  # Finds time of beginning of spe
            val10_8 = .1 * (min(v_spliced) - avg)  # Calculates 10% max
            val90_8 = 9 * val10_8  # Calculates 90% max
            tvals2_8 = np.linspace(t1_8, min_time, 5000)
            vvals2_8 = np.interp(tvals2_8, t8, v8)
            time10 = tvals2_8[np.argmin(np.abs(vvals2_8 - val10_8))]  # Finds time of point of 10% max
            time90 = tvals2_8[np.argmin(np.abs(vvals2_8 - val90_8))]  # Finds time of point of 90% max
            rise_time1090_8 = float(format(time90 - time10, '.2e'))  # Calculates 10-90 rise time
            x8 = 1

        if rise_time1090_1 <= 0 and x1 == 1:
            print(item, 'rt 1', rise_time1090_1)
            plt.plot(t1, v1)
            plt.plot(tvals2_1[np.argmin(np.abs(vvals2_1 - val10_1))], vvals2_1[np.argmin(np.abs(vvals2_1 - val10_1))],
                     'x')
            plt.plot(tvals2_1[np.argmin(np.abs(vvals2_1 - val90_1))], vvals2_1[np.argmin(np.abs(vvals2_1 - val90_1))],
                     'x')
            plt.show()
        elif rise_time1090_1 <= 0:
            print(item, 'rt 1')
            plt.plot(t1, v1)
            plt.show()

        if rise_time1090_2 <= 0 and x2 == 1:
            print(item, 'rt 2', rise_time1090_2)
            plt.plot(t2, v2)
            plt.plot(tvals2_2[np.argmin(np.abs(vvals2_2 - val10_2))], vvals2_2[np.argmin(np.abs(vvals2_2 - val10_2))],
                     'x')
            plt.plot(tvals2_2[np.argmin(np.abs(vvals2_2 - val90_2))], vvals2_2[np.argmin(np.abs(vvals2_2 - val90_2))],
                     'x')
            plt.show()
        elif rise_time1090_2 <= 0:
            print(item, 'rt 2')
            plt.plot(t2, v2)
            plt.show()

        if rise_time1090_4 <= 0 and x4 == 1:
            print(item, 'rt 4', rise_time1090_4)
            plt.plot(t4, v4)
            plt.plot(tvals2_4[np.argmin(np.abs(vvals2_4 - val10_4))], vvals2_4[np.argmin(np.abs(vvals2_4 - val10_4))],
                     'x')
            plt.plot(tvals2_4[np.argmin(np.abs(vvals2_4 - val90_4))], vvals2_4[np.argmin(np.abs(vvals2_4 - val90_4))],
                     'x')
            plt.show()
        elif rise_time1090_4 <= 0:
            print(item, 'rt 4')
            plt.plot(t4, v4)
            plt.show()

        if rise_time1090_8 <= 0 and x8 == 1:
            print(item, 'rt 8', rise_time1090_8)
            plt.plot(t8, v8)
            plt.plot(tvals2_8[np.argmin(np.abs(vvals2_8 - val10_8))], vvals2_8[np.argmin(np.abs(vvals2_8 - val10_8))],
                     'x')
            plt.plot(tvals2_8[np.argmin(np.abs(vvals2_8 - val90_8))], vvals2_8[np.argmin(np.abs(vvals2_8 - val90_8))],
                     'x')
            plt.show()
        elif rise_time1090_8 <= 0:
            print(item, 'rt 8')
            plt.plot(t8, v8)
            plt.show()
