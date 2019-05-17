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
    else:
        idx1 = int(.6 * len(x))
        idx2 = int(.9 * len(x))
    for j in range(idx1, idx2):
        y_sum += y[j]
    average = y_sum / (idx2 - idx1)
    return average


def calculate_charge(x, y):
    ysum = 0
    resistance = 50
    idx1 = np.inf
    idx2 = np.inf
    avg = calculate_average(x, y)
    min_val = np.amin(y)
    idx_min_val = np.where(y == min_val)
    time_min_val = x[idx_min_val]
    min_time = time_min_val[0]
    xvals = np.linspace(x[0], x[len(x) - 1], int(2e6))
    xvals1 = np.linspace(x[0], min_time, int(2e6))
    xvals2 = np.linspace(min_time, x[len(x) - 1], int(2e6))
    yvals = np.interp(xvals, x, y)
    yvals1 = np.interp(xvals1, x, y)
    yvals2 = np.interp(xvals2, x, y)

    yvals1_flip = np.flip(yvals1)
    difference_value1 = yvals1_flip - avg
    difference_value2 = yvals2 - avg
    for j in range(0, len(difference_value1) - 1):
        if difference_value1[j] >= 0:
            idx1 = len(difference_value1) - j
            break
    if idx1 == np.inf:
        idx1 = len(difference_value1) - np.argmin(np.abs(difference_value1))
    for j in range(0, len(difference_value2) - 1):
        if difference_value2[j] >= 0:
            idx2 = j
            break
    if idx2 == np.inf:
        idx2 = np.argmin(np.abs(difference_value2))
    x01 = xvals1[idx1]
    x02 = xvals2[idx2]
    diff_val1 = np.abs(xvals - x01)
    diff_val2 = np.abs(xvals - x02)
    index1 = np.argmin(diff_val1)
    index2 = np.argmin(diff_val2)
    x1 = xvals[index1]
    x2 = xvals[index2]
    delta_x = x2 - x1
    for j in range(index1.item(), index2.item()):
        ysum += yvals[j]
    area = delta_x * ysum / (index2 - index1)
    char = -1 * area / resistance
    return x1, x2, char


def calculate_amp(x, y):
    avg = calculate_average(x, y)
    min_val = np.amin(y)
    amp = avg - min_val
    return amp


def rise_time(x, y):
    min_val = np.amin(y)
    idx_min_val = np.where(y == min_val)
    time_min_val = x[idx_min_val]
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
    idx_min_val = np.where(y == min_val)
    time_min_val = x[idx_min_val]
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
