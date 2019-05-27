import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from read_waveform import read_waveform as rw
from write_waveform import write_waveform as ww

'''loops through all shifted files
    normalizes file
adds all v
divides by number of files
plots t and v'''


def average_waveform(start, end, dest_path, nhdr):
    data_file = Path(dest_path / 'd1_shifted')
    # save_file = Path(dest_path / 'plots')
    # vsum = 0
    for i in range(start, end + 1):
        file_name = 'D1--waveforms--%05d.txt' % i
        if os.path.isfile(data_file / file_name):
            print('Calculating average waveform: ', i)
            t, v, hdr = rw(data_file / file_name, nhdr)
            v = v / min(v)
            idx = np.where(t == 0)
            idx = int(idx[0])
            t = np.roll(t, -idx)
            v = np.roll(v, -idx)
            idx2 = np.where(t == min(t))
            if idx2 <= 3430:
                t = t[:3431]
                v = v[:3431]
                t = np.roll(t, -idx)
            print(idx2)
            print(t[3430])


average_waveform(0, 100, Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/full_bdw_no_nf/d1'),
                 5)
'''vsum += v
    yfinal = np.divide(ysum,(Nloops+1))
    header_name = "Average Waveform Shape"
    write_waveform(t,yfinal,writename,header_name)
    return (t,yfinal)

#Generate plot
def generate_average_shape_plot(data_date,numhead):
    (x,y) = determine_average_shape(data_date,numhead)
    fig = plt.figure(figsize=(6,4))
    plt.plot(x,y)
    plt.xlabel('Time')
    plt.ylabel('Ratio to Peak Height')
    plt.title('Average Wave Shape')
    plt.show()
    fig.savefig('G:/data/watchman/'+data_date+'_watchman_spe/d1/d1_histograms/average_shape.png',dpi = 300)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='determineaverageshape', description='determining and writing plot of average waveform shape')
    parser.add_argument('--data_date',type = str,help = 'date when data was gathered, YYYYMMDD', default = '20190516')
    parser.add_argument('--numhead',type=int,help='number of lines to ignore for header',default = 5)
    args = parser.parse_args()

    generate_average_shape_plot(args.data_date,args.numhead)'''