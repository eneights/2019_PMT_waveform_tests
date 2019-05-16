# FFT
T = N / fsps
df = 1 / T
dw = 2 * np.pi / T
H_FFT = np.fft.fft(h2)
MAG_H_FFT = [abs(IH) for IH in H_FFT[0:(len(h2) / 2 - 1)]]
f = np.fft.fftfreq(N) * N * df / 1.E9
f = f[:len(MAG_H_FFT)]
# print 'FFT of filter kernel'
# print '%d %d' % (len(f),len(MAG_H_FFT))
# plt.plot(f,MAG_H_FFT,'-o')
# plt.xscale('log')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import sys

Nloops = 100
show_plot = False

Y_FFT_AVG = []

for i in range(Nloops):
    fin = open('/media/tyler/Seagate Expansion Drive/20181212_watchman_spe/C2--waveforms--%05d.txt' % i)
    print
    i
    # Header
    for j in range(5):
        fin.readline()
        n = []
        x = []
        y = []
        ni = 0
    for line in fin:
        n.append(ni)
        x.append(float(line.split(',')[0]))
        y.append(float(line.split(',')[1]))
        ni += 1
    N = len(x)
    dt = x[1] - x[0]
    T = dt * N
    df = 1 / T
    dw = 2 * np.pi / T
    Y_FFT = np.fft.fft(y)
    f = np.fft.fftfreq(N) * N * df / 1.E9

    if (show_plot):
        # plt.plot(f)
        # plt.show()

        # plt.plot(f[0:N/2-1],abs(X)[0:N/2-1])
        # plt.yscale('log')
        # plt.show()

        plt.plot(f[0:N / 2 - 1], 20 * np.log10(abs(Y_FFT))[0:N / 2 - 1])
        plt.show()
        fin.close()

    if (i == 0):
        Y_FFT_AVG = Y_FFT.copy()
    else:
        Y_FFT_AVG = [(iy1 * i + iy2) / (i + 1) for iy1, iy2 in zip(Y_FFT_AVG, Y_FFT)]
        fin.close()

# print len(Y_FFT_AVG), len(Y_FFT)
MAG_Y_FFT_AVG = [abs(IY) for IY in Y_FFT_AVG[0:(N / 2 - 1)]]
# print MAG_Y_FFT_AVG

# sys.exit()

plt.plot(f[0:N / 2 - 1], 20 * np.log10(MAG_Y_FFT_AVG))
# plt.xscale('log')
plt.show()

plt.plot(np.arange(len(MAG_Y_FFT_AVG)), 20 * np.log10(MAG_Y_FFT_AVG))
# plt.xscale('log')
plt.show()