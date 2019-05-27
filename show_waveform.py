from pathlib import Path
import matplotlib.pyplot as plt
from read_waveform import read_waveform as rw

path = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/full_bdw_no_nf/d1/d1_raw')
file = 'D1--waveforms--51589.txt'
file_name = str(path / file)
nhdr = 5

t, v, hdr = rw(file_name, nhdr)
print("\nHeader:\n\n" + str(hdr))
plt.plot(t, v)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.show()
