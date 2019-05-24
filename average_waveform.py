import numpy as np
import matplotlib.pyplot as plt
from readwaveform import read_waveform as rw
from writewaveform import write_waveform
import os

#Determining average shape
def determine_average_shape(data_date,numhead):
    Nloops = len(os.listdir('G:/data/watchman/'+data_date+'_watchman_spe/d1/d1_normalized'))
    writename = 'G:/data/watchman/'+data_date+'_watchman_spe/d1/d1_histograms/average_shape.txt'
    for i in range(Nloops):
        print(i)
        filename = 'G:/data/watchman/'+data_date+'_watchman_spe/d1/d1_normalized/D1--waveforms--%05d.txt' % i
        (t,y,_) = rw(filename,numhead)
        if i == 0:
            ysum = y
        else:
            ysum = np.add(ysum,y)
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

    generate_average_shape_plot(args.data_date,args.numhead)