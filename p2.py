from scipy import signal


def p2(start, end, numtaps, fc, fsps,):
    lowpass1 = signal.firwin(numtaps, cutoff=2. * fc / fsps, window='blackman')
    lowpass2 = signal.firwin(numtaps, cutoff=wc/np.pi, window='blackman')
    lowpass4 = signal.firwin(numtaps, cutoff=wc/np.pi, window='blackman')
    lowpass8 = signal.firwin(numtaps, cutoff=wc/np.pi, window='blackman')