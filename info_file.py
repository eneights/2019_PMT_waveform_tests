import datetime
from pathlib import Path
from read_waveform import read_waveform as rw


def info_file(acq_date_time, source_path, dest_path, pmt_hv, gain, offset, trig_delay, amp, fsps, band, nfilter, r):
    now = datetime.datetime.now()
    file_name = 'info.txt'
    file = dest_path / file_name
    myfile = open(file, 'w')
    myfile.write('Data acquisition,' + str(acq_date_time))
    myfile.write('\nData processing,' + str(now))
    myfile.write('\nSource data,' + str(source_path))
    myfile.write('\nDestination data,' + str(dest_path))
    myfile.write('\nPMT HV (V),' + str(pmt_hv))
    myfile.write('\nNominal gain,' + str(gain))
    myfile.write('\nDG 535 offset,' + str(offset))
    myfile.write('\nDG 535 trigger delay (ns),' + str(trig_delay))
    myfile.write('\nDG 535 amplitude (V),' + str(amp))
    myfile.write('\nOscilloscope sample rate (Hz),' + str(fsps))
    myfile.write('\nOscilloscope bandwidth (Hz),' + str(band))
    myfile.write('\nOscilloscope noise filter (bits),' + str(nfilter))
    myfile.write('\nOscilloscope resistance (ohms),' + str(r))
    myfile.close()
