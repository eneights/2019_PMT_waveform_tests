from functions import *

gen_path = Path(r'/Volumes/TOSHIBA EXT/data/watchman')
save_path = Path(str(gen_path / '%08d_watchman_spe/waveforms/%s') % (20190513, 'full_bdw_no_nf'))
data_path = Path(save_path / 'd0')

filename = data_path / 'C2--waveforms--00000.txt'

t, v, hdr = rw(filename, 5)

v_new = np.sort(v)

for item in v_new:
    print(item)
