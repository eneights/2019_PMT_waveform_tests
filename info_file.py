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


if __name__ == '__main__':
    data = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/full_bandwidth_no_noise_filter/d0')
    save = Path(r'/Volumes/TOSHIBA EXT/data/watchman/20190513_watchman_spe/waveforms/full_bandwidth_no_noise_filter/d1')
    current = datetime.datetime.now()
    import argparse
    parser = argparse.ArgumentParser(prog="info file", description="create info file for p1.")
    parser.add_argument("--acq_date_time", type=str, help="date & time of data acquisition", default=current)
    parser.add_argument("--source_path", type=str, help="path to original data", default=data)
    parser.add_argument("--dest_path", type=str, help="path to d1 folder", default=save)
    parser.add_argument("--pmt_hv", type=int, help="voltage of PMT", default=1800)
    parser.add_argument("--gain", type=int, help="gain of PMT", default=1e7)
    parser.add_argument("--offset", type=int, help="offset of pulse generator", default=0)
    parser.add_argument("--trig_delay", type=float, help="delay of pulse generator trigger", default=9.)
    parser.add_argument("--amp", type=float, help="amplitude of pulse generator", default=3.5)
    parser.add_argument("--fsps", type=float, help='samples per second (Hz)', default=20000000000.)
    parser.add_argument("--band", type=int, help="bandwidth of oscilloscope", default=0)
    parser.add_argument("--nfilter", type=float, help="noise filter on oscilloscope", default=0)
    parser.add_argument("--r", type=int, help='resistance in ohms', default=50)
    args = parser.parse_args()

    info_file(args.acq_date_time, args.source_path, args.dest_path, args.pmt_hv, args.gain, args.offset,
              args.trig_delay, args.amp, args.fsps, args.band, args.nfilter, args.r)
