from scipy import signal
from functions import *


def p2(start, end, numtaps, fc, fsps, nhdr):

    lowpass1 = signal.firwin(numtaps, cutoff=2. * fc / fsps, window='blackman')
    lowpass2 = signal.firwin(numtaps, cutoff=fc / fsps, window='blackman')
    lowpass4 = signal.firwin(numtaps, cutoff=fc / (2. * fsps), window='blackman')
    lowpass8 = signal.firwin(numtaps, cutoff=fc / (4. * fsps), window='blackman')

    file_name =

    t, v, hdr = rw(file_name, nhdr)
    v1 = signal.filtfilt(lowpass1, 1.0, v)
    v2 = signal.filtfilt(lowpass2, 1.0, v)
    v4 = signal.filtfilt(lowpass4, 1.0, v)
    v8 = signal.filtfilt(lowpass8, 1.0, v)
