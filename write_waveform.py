from read_waveform import read_waveform as rw


def write_waveform(x, y, file_name, hdr):
    myfile = open(file_name, 'w')           # opens file
    for entry in str(hdr):
        myfile.write(entry)                 # writes header in file
    for ix, iy in zip(x, y):
        line = '%.7E,%f\n' % (ix, iy)       # formats line
        myfile.write(line)                  # writes line in file
    myfile.close()


if __name__ == '__main__':
    header_string = "LECROYWaveRunner,20113,Waveform\nSegments,1,SegmentSize,4002\n\n\nTime,Ampl"
    import argparse
    parser = argparse.ArgumentParser(prog="write waveform", description="write a waveform datafile.")
    parser.add_argument("--hdr", type=str, help='header string for the output file', default=header_string)
    parser.add_argument("--file_name", type=str, help="filename", default="./C2--waveforms--00000.txt")
    args = parser.parse_args()

    t, v, header = rw(args.file_name, 5)
    write_waveform(t, v, args.file_name, header)
