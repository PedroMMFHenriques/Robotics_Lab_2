# Robotics lab 2, Control
"""
    Authors:
    - Diogo Bento, 93045
    - Francisco Carrilho, 93062
    - João Batista, 93102
    Date last modified: 31/01/2022
"""

from math import dist
from typing import Counter
import numpy as np
from scipy.signal import butter, lfilter, freqz, filtfilt
import matplotlib.pyplot as plt
import csv

def joint(xref, yref, theta_ref, x, y, index, values, counter):
	diff_orientation = 0

	final_index = len(values)
	if index<len(values)- 10:
		final_index= index+10
	j=index+1

	global control
	control=0
	for i in range(index,final_index):
		diff=np.abs(theta_ref[i]-theta_ref[index])
		if diff > (diff_orientation+np.pi/8):
			diff_orientation=diff
			j = i
			control=1
			counter=counter+1
			break
	if control==0:
		dist_shortest = np.inf

		final_index = len(values)-1

		for i in range(index,final_index):
			dist = np.sqrt((x-xref[i])**2 + (y-yref[i])**2)
			if dist < dist_shortest:
				dist_shortest = dist
				j = i
	return j

def read_csv_file(filename):
	file = open(filename)

	csvreader = csv.reader(file)
	
	values = []
	for row in csvreader:
		row = [float(i) for i in row]
		values.append(row)

	return values

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff= cutoff/nyq
    b, a = butter(order, normal_cutoff, btype = 'low', analog = False)
    y = filtfilt(b, a, data)
    return y

def control_it(i, j, time, xref, yref, theta_ref, x, y, values, we, theta, v_no_filter, v, Kv, ws_no_filter, Ks, Ki, ws, h, phi, L, counter, ws_filtered,
	v_filtered):
	time.append(time[i-1] + h)
	j=joint(xref, yref, theta_ref, x[i-1], y[i-1], j, values, counter)
	if control==0:
		if j >= len(values) - 2:
			k = len(values) - 1
		else:
			k = j + 2
	else:
		k=j; 

	we[0] = xref[k] - x[i-1]
	we[1] = yref[k] - y[i-1]
	we[2] = theta_ref[k] - theta[i-1]

	be = (np.dot(np.array([[np.cos(theta[i-1]),np.sin(theta[i-1]), 0], [-np.sin(theta[i-1]), np.cos(theta[i-1]), 0 ],[0 , 0, 1]]), we))
	
	v_no_filter.append(Kv*be[0])
	if i < 10:
		v_filtered.append(v_no_filter[i-1])
	else :
		v_filtered = butter_lowpass_filter(v_no_filter, 0.03, 1/h, 2)

	v.append(v_filtered[i-1])

	if v[i-1] > 5.55:
		v[i-1] = 5.55
	if v[i-1] < 1:
		v[i-1] = 1
	
	ws_no_filter.append(Ks*be[2] + Ki*be[1])
	if i < 10:
		ws_filtered.append(ws_no_filter[i-1])
	else: 
		ws_filtered = butter_lowpass_filter(ws_no_filter, 0.3, 1/h, 2)
	ws.append(ws_filtered[i-1])  
	delta = np.dot(np.array([[np.cos(theta[i-1]), 0], [np.sin(theta[i-1]), 0 ],[np.tan(phi[i-1])/L , 0], [0, 1]]), np.transpose(np.array([v[i-1], ws[i-1]])))
	x.append(x[i-1] + h*delta[0])
	y.append(y[i-1] + h*delta[1])
	theta.append(theta[i-1] + h*delta[2])
	phi.append(np.pi/7*np.tanh((phi[i-1] + h*delta[3])))
	return v[i-1], ws[i-1], j, phi[i-1], theta[i-1]