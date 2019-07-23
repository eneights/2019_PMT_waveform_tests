from functions import *

gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (20190513, 'full_bdw_no_nf'))
data_path = Path(save_path / 'd0')

filename = data_path / 'C2--waveforms--00000.txt'

t, v, hdr = rw(filename, 5)

v_new = np.sort(v)

diff_array = np.array([])

for i in range(len(v_new)):
    if i >= 1:
        diff = v_new[i] - v_new[i - 1]
        diff_array = np.append(diff_array, diff)

# diff_array[diff_array == 0] = np.inf

diff_new = np.sort(diff_array)

plt.hist(diff_new[diff_new <= 0.00001], 100)
plt.title('Difference Between 20 Gsps Oscilloscope Voltages')
plt.xlabel('Difference Between Voltages (V)')
plt.show()
