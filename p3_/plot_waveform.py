from functions import *

gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/full_bdw_no_nf/d3')
folder = str('double_spe/no_delay/digitized')
# file_num = 3
# file_name = str(gen_path / folder / 'D3--waveforms--%05d.txt') % file_num
file_name = str(gen_path / folder / 'D3--waveforms--75373--39262.txt')

t, v, hdr = rw(file_name, 5)
print("\nHeader:\n\n" + str(hdr))
t_step = t[1] - t[0]
plt.scatter(t, v, s=15)
plt.xlim([min(t) - 10 * t_step, max(t) + 10 * t_step])
plt.plot(t, v)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.show()
