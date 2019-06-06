from functions import *

gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/full_bdw_no_nf/d2')
folder = 'filter8'
file_num = 229
file_name = str(gen_path / folder / 'D2--waveforms--%05d.txt') % file_num

t, v, hdr = rw(file_name, 5)
print("\nHeader:\n\n" + str(hdr))
plt.plot(t, v)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.show()
