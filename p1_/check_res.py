from functions import *

gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
# save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (20190513, 'full_bdw_no_nf'))
# data_path = Path(save_path / 'd0')
data_path = Path(str(gen_path / '%08d_watchman_spe') % 20190723)

filename = data_path / 'C2--waveforms--00001.txt'

t, v, hdr = rw(filename, 5)

v_new = np.sort(v)

diff_array = np.array([])

for i in range(len(v_new)):
    if i >= 1:
        diff = v_new[i] - v_new[i - 1]
        diff_array = np.append(diff_array, diff)

# diff_array[diff_array == 0] = np.inf

diff_new = np.sort(diff_array)

n, bins, patches = plt.hist(diff_new[diff_new <= 0.00001], 50)

voltages_array = np.array([])
idx = np.where(n > 0)
idx_array = idx[0]

for item in idx_array:
    voltages_array = np.append(voltages_array, bins[item])

x = 0
res_val = 0
for i in range(len(voltages_array)):
    if i > 0:
        x += 1
        res_val += voltages_array[i] - voltages_array[i - 1]

avg_res_val = res_val / x
avg_res_val = float(format(avg_res_val, '.5e'))
plt.yscale('log')
plt.title('Resolution of 20 Gsps Oscilloscope\nQuantization: ' + str(avg_res_val) + ' V')
plt.xlabel('Difference Between Voltages (V)')
plt.show()
