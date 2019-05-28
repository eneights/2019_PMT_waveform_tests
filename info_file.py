import datetime


# Creates info file
def info_file(acq_date_time, source_path, dest_path, pmt_hv, gain, offset, trig_delay, amp, fsps, band, nfilter, r):
    now = datetime.datetime.now()
    file_name = 'info.txt'
    file = dest_path / file_name
    myfile = open(file, 'w')
    myfile.write('Data acquisition,' + str(acq_date_time))              # date & time of raw data from previous info
                                                                        # file
    myfile.write('\nData processing,' + str(now))                       # current date & time
    myfile.write('\nSource data,' + str(source_path))                   # path to source data
    myfile.write('\nDestination data,' + str(dest_path))                # path to folder of current data
    myfile.write('\nPMT HV (V),' + str(pmt_hv))                         # voltage of PMT from previous info file
    myfile.write('\nNominal gain,' + str(gain))                         # gain of PMT from previous info file
    myfile.write('\nDG 535 offset,' + str(offset))                      # offset of pulse generator from previous info
                                                                        # file
    myfile.write('\nDG 535 trigger delay (ns),' + str(trig_delay))      # trigger delay of pulse generator from previous
                                                                        # info file
    myfile.write('\nDG 535 amplitude (V),' + str(amp))                  # amplitude of pulse generator from previous
                                                                        # info file
    myfile.write('\nOscilloscope sample rate (Hz),' + str(fsps))        # sample rate of oscilloscope from previous info
                                                                        # file
    myfile.write('\nOscilloscope bandwidth (Hz),' + str(band))          # bandwidth of oscilloscope from previous info
                                                                        # file
    myfile.write('\nOscilloscope noise filter (bits),' + str(nfilter))  # oscilloscope noise filter from previous info
                                                                        # file
    myfile.write('\nOscilloscope resistance (ohms),' + str(r))          # resistance of oscilloscope from previous info
                                                                        # file
    myfile.close()
