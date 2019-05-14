import numpy as np
import matplotlib.pyplot as plt

flist = open('./a1_data/a1_sel.txt')

for l1 in flist:
    i = int(l1.split(' ')[0])
    fin = open('/media/tyler/Seagate Expansion Drive/20181220_watchman_spe_filter/l1/%05d.txt' % i)
    print i
    # Header
    for j in range(5):
        fin.readline()
    x = []
    y = []
    for line in fin:
        x.append(float(line.split(',')[0]))
        y.append(float(line.split(',')[1]))
    plt.plot(x,y)
    plt.ylim(-0.0250,0.005)
    plt.show()
    fin.close()
flist.close()