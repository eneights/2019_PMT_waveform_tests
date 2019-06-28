from functions import *

gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/full_bdw_no_nf/d3')
folder = str('double_spe/rt_4/no_delay/digitized_250_Msps')
# folder = str('single_spe/rt_8/digitized_125_Msps')
# folder = str('rt_2/raw')
# file_num = 4280
# file_name = str(gen_path / folder / 'D3--waveforms--%05d.txt') % file_num
# file_name = str(gen_path / folder / 'D3--waveforms--00625--60806.txt')

double_file_array = np.array([])

print('Checking existing files...')
for filename in os.listdir(gen_path / folder):
    print(filename, 'is a file')
    files_added = filename[15:27]
    double_file_array = np.append(double_file_array, files_added)

'''print('Checking existing files...')
for filename in os.listdir(gen_path / folder):
    print(filename, 'is a file')
    files_added = filename[15:20]
    double_file_array = np.append(double_file_array, files_added)'''

for item in double_file_array:
    file_name = str(gen_path / folder / 'D3--waveforms--%s.txt') % item
    t, v, hdr = rw(file_name, 5)
    plt.plot(t, v)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.show()
    '''fwhm = calculate_fwhm(t, v, -50)
    if fwhm == -1:
        print('FWHM is', fwhm)'''


# t, v, hdr = rw(file_name, 5)
# print("\nHeader:\n\n" + str(hdr))
# t_step = t[1] - t[0]
# plt.scatter(t, v, s=15)
# plt.xlim([min(t) - 10 * t_step, max(t) + 10 * t_step])
# plt.plot(t, v)
# plt.xlabel('Time (s)')
# plt.ylabel('Voltage (V)')
# plt.show()
