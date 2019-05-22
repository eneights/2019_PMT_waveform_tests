from read_waveform import read_waveform as rw
import matplotlib.pyplot as plt
from pathlib import Path

file_name = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/200MHz_3bit/d0'
                 r'/C2--waveforms--00002.txt')
t, v, hdr = rw(file_name, 5)
print("\nHeader:\n\n" + str(hdr))
plt.plot(t, v)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.show()

vsum = 0
for i in range(0, len(v) - 1):
    vsum += v[i]
baseline = vsum / len(v)

print(baseline)