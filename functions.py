import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal


def calculate_average(x, y):
    y_sum = 0
    idx = np.abs(x - 0).argmin()
    if idx > len(x) / 2:
        idx1 = int(.1 * len(x))
        idx2 = int(.4 * len(x))
    if idx <= len(x) / 2:
        idx1 = int(.6 * len(x))
        idx2 = int(.9 * len(x))
    for j in range(idx1, idx2):
        y_sum += y[j]
    average = y_sum / (idx2 - idx1)
    return average


def calculate_charge(x, y):
    ysum = 0
    resistance = 50
    avg = calculate_average(x, y)
    xvals = np.linspace(x[0], x[len(x) - 1], int(2e6))
    yvals = np.interp(xvals, x, y)
    difference_value1 = yvals - avg
    difference_value2 = np.abs(xvals - 0)
    idx = np.argmin(difference_value2)
    diff_split = np.split(difference_value1, [idx])
    diff_split1 = np.flip(diff_split[0])
    diff_split2 = diff_split[1]
    for j in range(0, len(diff_split1)):
        if diff_split1[j] > 0:
            idx1 = len(diff_split1) - j
            break
        elif yvals[j] == 0:
            idx1 = len(yvals) - j
            break
        else:
            difference_value3 = np.abs(yvals - 0)
            diff_split_2 = np.split(difference_value3, [idx])
            diff_split3 = np.flip(diff_split_2[0])
            idx1 = np.argmin(diff_split3)
            break
    for j in range(0, len(diff_split2)):
        if diff_split2[j] > 0:
            idx2 = idx + j
            break
        elif yvals[j] == 0:
            idx2 = len(yvals) - j
            break
        else:
            difference_value4 = np.abs(yvals - 0)
            diff_split_3 = np.split(difference_value4, [idx])
            diff_split4 = np.flip(diff_split_3[1])
            idx2 = np.argmin(diff_split4)
            break
    x1 = xvals[idx1]
    x2 = xvals[idx2]
    delta_x = x2 - x1
    for j in range(idx1, idx2):
        ysum += yvals[j]
    area = delta_x * ysum / (idx2 - idx1)
    char = -1 * area / resistance
    return x1, x2, char


def calculate_amp(x, y):
    avg = calculate_average(x, y)
    min_val = np.amin(y)
    amp = avg - min_val
    return amp


def rise_time(x, y):
    min_val = np.amin(y)
    ind_min_val = np.where(y == min_val)
    time_min_val = x[ind_min_val]
    min_time = time_min_val[0]
    avg = calculate_average(x, y)
    x1, x2, char = calculate_charge(x, y)
    val10 = .1 * (min_val - avg)
    val20 = 2 * val10
    val80 = 8 * val10
    val90 = 9 * val10
    xvals = np.linspace(x1, min_time, int(2e6))
    yvals = np.interp(xvals, x, y)
    difference_value10 = np.abs(yvals - val10)
    difference_value20 = np.abs(yvals - val20)
    difference_value80 = np.abs(yvals - val80)
    difference_value90 = np.abs(yvals - val90)
    index10 = np.argmin(difference_value10)
    index20 = np.argmin(difference_value20)
    index80 = np.argmin(difference_value80)
    index90 = np.argmin(difference_value90)
    time10 = xvals[index10]
    time20 = xvals[index20]
    time80 = xvals[index80]
    time90 = xvals[index90]
    rise_time1090 = time90 - time10
    rise_time2080 = time80 - time20
    rise_time1090 = float(format(rise_time1090, '.2e'))
    rise_time2080 = float(format(rise_time2080, '.2e'))
    return rise_time1090, rise_time2080


def fall_time(x, y):
    min_val = np.amin(y)
    ind_min_val = np.where(y == min_val)
    time_min_val = x[ind_min_val]
    min_time = time_min_val[0]
    avg = calculate_average(x, y)
    x1, x2, char = calculate_charge(x, y)
    val10 = .1 * (min_val - avg)
    val20 = 2 * val10
    val80 = 8 * val10
    val90 = 9 * val10
    xvals = np.linspace(min_time, x2, int(2e6))
    yvals = np.interp(xvals, x, y)
    difference_value10 = np.abs(yvals - val10)
    difference_value20 = np.abs(yvals - val20)
    difference_value80 = np.abs(yvals - val80)
    difference_value90 = np.abs(yvals - val90)
    index10 = np.argmin(difference_value10)
    index20 = np.argmin(difference_value20)
    index80 = np.argmin(difference_value80)
    index90 = np.argmin(difference_value90)
    time10 = xvals[index10]
    time20 = xvals[index20]
    time80 = xvals[index80]
    time90 = xvals[index90]
    fall_time1090 = time10 - time90
    fall_time2080 = time20 - time80
    fall_time1090 = float(format(fall_time1090, '.2e'))
    fall_time2080 = float(format(fall_time2080, '.2e'))
    return fall_time1090, fall_time2080
