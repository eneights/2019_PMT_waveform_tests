from functions import *

gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/full_bdw_no_nf/d3')
folder = str('double_spe/raw')
# file_num = 3
# file_name = str(gen_path / folder / 'D3--waveforms--%05d.txt') % file_num
file_name = str(gen_path / folder / 'D3--waveforms--69653--.txt')

t, v, hdr = rw(file_name, 5)
print("\nHeader:\n\n" + str(hdr))
plt.plot(t, v)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.show()
