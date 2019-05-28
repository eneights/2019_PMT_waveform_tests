# average_waveform.py

Using shifted data (csv files with time and amplitude columns, shifted so that baseline noise level = 0 and at time = 0, voltage = 50% max), calculates the average waveform of a spe




# functions.py

Functions that are used in p1.py, average_waveform.py, p1_sort.py, and plot_waveform.py

rw: read waveform; reads csv file and returns times, voltages & header

ww: write waveform; writes a csv file with header and time & voltage columns

calculate_average: calculates baseline noise level

subtract_time: shifts spes so that baseline noise level = 0 and at time = 0, voltage = 50% max

calculate_charge: calculates charge of spe

calculate_amp: calculates amplitude of spe

calculate_fwhm: calculates FWHM of spe

rise_time: calculates 10-90 & 20-80 rise times of spe

fall_time: calculates 10-90 & 20-80 fall times of spe

calculate_times: calculates 10%, 20%, 80%, and 90% jitter of spe

save_calculations: creates text file with time of beginning of spe, time of end of spe, charge, amplitude, fwhm, 10-90 & 20-80 rise times, 10-90 & 20-80 fall times, and 10%, 20%, 80% & 90% jitter

write_hist_data: creates text file with data from array

make_arrays: creates arrays of beginning & end times of spe waveform, charge, amplitude, fwhm, 10-90 & 20-80 rise times, 10-90 & 20-80 fall times, and 10%, 20%, 80% & 90% jitter

plot_histogram: creates histogram using values from array

remove_outliers: removes outliers from array




# info_file.py

Used in p1.py

Creates d1 info file




# p1.py

Using raw data from oscilloscope (csv files with time and amplitude columns), creates histograms of charge, amplitude, FWHM, 10-90 & 20-80 rise times, 10-90 & 20-80 fall times, and 10%, 20%, 80% & 90% jitter with and without outliers

Requires the input of a 2 column csv info file with the following information in the second column of each row:
Date & time of data acquisition (YYYY-MM-DD HH:MM)
Path to folder with raw data
Number of header lines in raw data files
Baseline of raw data in V
Voltage of PMT in V
Gain of PMT
Offset of pulse generator in V
Trigger delay of pulse generator in ns
Amplitude of pulse generator in V
Sample rate of oscilloscope in Hz
Bandwidth of oscilloscope in Hz
Noise filter of oscilloscope in bits
Resistance of oscilloscope in ohms

Can specify file numbers to start and end at (default is 0 and 99999)




# p1_sort.py

Used in p1.py

Sorts raw data from oscilloscope into spes, non-apes, and maybe spes
Will prompt user to manually sort some files




# plot_waveform.py

Plots a waveform for the user to view

Can input folder name and file number of the waveform