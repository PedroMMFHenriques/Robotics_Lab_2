from math import dist
from typing import Counter
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import csv

def joint(xref, yref, theta_ref, x, y, index):
	diff_orientation = 0
	#if index + 10 > len(values):
	final_index = len(values)
	if index<len(values)-40:
		final_index= index+40
	j=index+1
	#else:
	#	final_index = index + 10
	global control
	global counter
	control=0
	for i in range(index,final_index):
		diff=np.abs(theta_ref[i]-theta_ref[index])
		if diff > (diff_orientation+1.7):
			diff_orientation=diff
			j = i
			control=1
			counter=counter+1
			break
	if control==0:
		dist_shortest = np.inf
		#if index + 10 > len(values):
		final_index = len(values)-1
		#else:
		#	final_index = index + 10
		for i in range(index,final_index):
			dist = np.sqrt((x-xref[i])**2 + (y-yref[i])**2)
			if dist < dist_shortest:
				dist_shortest = dist
				j = i
	return j

def corner_cutting(theta_ref, theta, index):
	diff_orientation = 0
	#if index + 10 > len(values):
	final_index = len(values)
	if index<len(values)-10:
		final_index= index+10
	j=index+1
	#else:
	#	final_index = index + 10
	for i in range(index,final_index):
		diff=np.abs(theta_ref[i]-theta_ref[index])
		if diff > (diff_orientation+1.5):
			diff_orientation=diff
			j = i
			break
	return j

def check_point(xref, yref, x, y, index):
	dist_shortest = np.inf
	#if index + 10 > len(values):
	final_index = len(values)-1
	#else:
	#	final_index = index + 10
	for i in range(index,final_index):
		dist = np.sqrt((x-xref[i])**2 + (y-yref[i])**2)
		if dist < dist_shortest:
			dist_shortest = dist
			j = i
	return j

def moving_average(x,index):
	average_length=10
	if index<average_length:
		y=0
		for k in range(0, index):
			y= x[k]+y
		y=y/(index+1)
	else:
		y=0
		for k in range(index-average_length+1, index):
			y= x[k]+y
		y=y/average_length
	return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def read_csv_file(filename):
	file = open(filename)

	csvreader = csv.reader(file)
	
	values = []
	for row in csvreader:
		values.append(row)

	return values


def lpf(x, omega_c, T):
    """Implement a first-order low-pass filter.
    
    The input data is x, the filter's cutoff frequency is omega_c 
    [rad/s] and the sample time is T [s].  The output is y.
    """
    y = x
    alpha = (2-T*omega_c)/(2+T*omega_c)
    beta = T*omega_c/(2+T*omega_c)
    for k in range(1, len(x)):
        y[k] = alpha*y[k-1] + beta*(x[k]+x[k-1])
    return y


values = read_csv_file('trajectory_points.csv')

xref = [int(i) for i in np.array(values)[:, 0]]
yref = [int(i) for i in np.array(values)[:, 1]]
theta_ref = [float(i) for i in np.array(values)[:, 2]]

xref = np.array(xref)*5.073825503/100
yref = np.array(yref)*5.073825503/100
theta_ref = np.array(theta_ref)


L = 2.2
#h - perido de sampling
h = 0.1 
#kv - constante de erro em x controla a velocidade
Kv = 0.03 * 30
#ks - constante de erro de orientcao (theta)
Ks = 100
#ki - constante de erro em y 
Ki = 10
#cenas para o filtro
omega_c = 2.0*np.pi*(0+0.1)/2.0
time = []
theta = []
phi = []
x = []
y = []
#be - erros no frame do carro
#be = []
# Setting standard filter requirements.
order = 6
fs = 1/h      
cutoff = 0.05
#b, a = butter_lowpass(cutoff, fs, order)

time.append(0)
theta.append(theta_ref[0])
phi.append(0)
#we - erros no frame do mundo
we = np.zeros(3)
x.append(xref[0])
y.append(yref[0])
v_no_filter = []
ws_no_filter = []
ws = []
v = []
i = 1
j = 0
k = 0
counter=0
print(len(values))
while j < len(values) - 2:
	time.append(time[i-1] + h)
	#j = check_point(xref, yref, x[i-1], y[i-1], j) 
	#j= corner_cutting(theta_ref, theta[i-1], j)
	j=joint(xref, yref, theta_ref, x[i-1], y[i-1], j)
	print(j)
	if control==0:
		if j >= len(values)-10:
			k = len(values) - 1
		else:
			k = j +10
	else:
		k=j; 

	#k = j
	#print(k + 2)
	we[0] = xref[k] - x[i-1]
	we[1] = yref[k] - y[i-1]
	we[2] = theta_ref[k] - theta[i-1]

	be = (np.dot(np.array([[np.cos(theta[i-1]),np.sin(theta[i-1]), 0], [-np.sin(theta[i-1]), np.cos(theta[i-1]), 0 ],[0 , 0, 1]]), we))
	v.append(Kv*be[0])
	
	#v = lpf(v_no_filter, omega_c,h)
	if v[i-1] > 5.55:
		v[i-1] = 5.55
	if v[i-1] < 1:
		v[i-1] = 1
	
	#ws.append(Ks*be[i-1][2] + Ki*be[i-1][1])
	ws_no_filter.append(Ks*be[2] + Ki*be[1])
	#if ws_no_filter[i-1] > np.pi:
	#	ws_no_filter[i-1] = np.pi
	#if ws_no_filter[i-1] < -np.pi/2:
	#	ws_no_filter[i-1] = -np.pi/2
	#ws = lpf(ws_no_filter,omega_c,h)
	#ws = butter_lowpass_filter(ws_no_filter, cutoff, fs, order)
	#if i-1==0:
		#ws.append(ws_no_filter[i-1])        
	#else: 
		#d=moving_average(ws_no_filter,i-1)
		#ws.append(d)
	ws.append(ws_no_filter[i-1])  
	delta = np.dot(np.array([[np.cos(theta[i-1]), 0], [np.sin(theta[i-1]), 0 ],[np.tan(phi[i-1])/L , 0], [0, 1]]), np.transpose(np.array([v[i-1], ws[i-1]])))
	x.append(x[i-1] + h*delta[0])
	y.append(y[i-1] + h*delta[1])
	theta.append(theta[i-1] + h*delta[2])
	print(h*delta[2])
	#phi.append(phi[i-1] + h*delta[3])
	#phi.append((np.pi/8)*np.tanh(phi[i-1] + h*delta[3]))
	if phi[i-1] + h*delta[3] > np.pi/8:
		phi.append(np.pi/8)
	elif phi[i-1] + h*delta[3] < -(np.pi/8):
		phi.append(-(np.pi/8))
	else:
		phi.append(phi[i-1] + h*delta[3])
	i += 1
print(counter)
plot1 = plt.figure(1)
plt.gca().invert_yaxis()
plt.plot(xref, yref, label="Trajetoria")
plt.plot(x,y, label="Carro")
plot2 = plt.figure("Orientacao")
plt.plot(time,theta)
#plt.plot(time,theta_ref) 
plot3 = plt.figure("Velocidade")
plt.plot(time[1:], v)
plot4 = plt.figure("Angulo de Steering")
plt.plot(time, phi)
plot5 = plt.figure("Velocidade de Steering")
plt.plot(time[1:], ws)
plt.show()