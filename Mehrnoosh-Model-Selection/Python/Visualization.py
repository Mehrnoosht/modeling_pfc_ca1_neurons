#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:31:46 2017

@author: Mehrnoosh
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as scis
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal as scis

from src.data_processing import (get_interpolated_position_dataframe,
                                 get_LFP_dataframe,
                                 get_spike_indicator_dataframe,
                                 make_epochs_dataframe, make_neuron_dataframe,
                                 make_tetrode_dataframe)
from src.parameters import ANIMALS, N_DAYS

days = range(1, N_DAYS + 1)
epoch_info = make_epochs_dataframe(ANIMALS, days)
tetrode_info = make_tetrode_dataframe(ANIMALS)

epoch_key = ('HPa', 6, 2)
tetrode_key = ('HPa', 6, 2, 5)
neuron_info = make_neuron_dataframe(ANIMALS)
neuron_key = ('HPa', 6, 2, 5, 2)

spike = get_spike_indicator_dataframe(neuron_key, ANIMALS)
linear_position = get_interpolated_position_dataframe(epoch_key, ANIMALS)[
    'linear_position']
x_pos = get_interpolated_position_dataframe(epoch_key, ANIMALS)['x_position']
y_pos = get_interpolated_position_dataframe(epoch_key, ANIMALS)['y_position']
speed = get_interpolated_position_dataframe(epoch_key, ANIMALS)['speed']
head_direction = get_interpolated_position_dataframe(epoch_key, ANIMALS)[
    'head_direction']

spike_position = spike.assign(
    linear_pos=linear_position, x_position=x_pos,
    y_position=y_pos, speed=speed, head_direction=head_direction)
spike_pos = spike_position[spike['is_spike'] == 1]


eeg = get_LFP_dataframe(tetrode_key, ANIMALS)
eeg.to_csv('eeg.csv', sep=',')

# Visualizing Spike Time

spike_time = spike[spike.values == 1]['is_spike']
y_spike = np.ones(spike_time.shape)
plt.figure(1)
plt.scatter(spike_time.index, y_spike)
plt.xlabel('Spike_Time')
plt.title('Tetrode:5  Neuron:2')
plt.show()

# Visualizing Interspike(ISI)

isi = pd.DataFrame(np.diff(spike_time.index))
y_isi = np.ones(isi.shape)

plt.figure(2)
plt.scatter(isi, y_isi)
plt.xlabel('InterSpike')
plt.title('Tetrode:5  Neuron:2')

plt.show()


# Spike Histogram

idx = spike.index[int(len(spike.index) / 3):int(2 * len(spike.index) / 3)]
idx = np.array(idx)
#rand_spike = spike.reindex(np.random.permutation(spike.index))
#end = int(len(rand_spike)/10000)
#idx =  np.array([rand_spike.index])[:10+1]
mx = np.amax(idx)
mn = np.amin(idx)
delta = mx - mn
bins = np.linspace(mn, mx, int(delta) * 100)
#rate_spike = np.histogram(spike_time.index, bins=int(delta)*100)[0]
# plt.bar(bins,rate_spike)

plt.figure(3)
plt.hist(spike_pos.index, bins=int(delta) * 100)
plt.xlim(mn, mx)
plt.xlabel('time[s]')
plt.ylabel('Number of Spikes')
plt.title('Tetrode:5  Neuron:2')
plt.show()


# ISI Histogram
idx_isi = np.array([isi.values])
mx_isi = np.amax(idx_isi)
mn_isi = np.amin(idx_isi)
delta_isi = mx_isi - mn_isi
bins_isi = np.linspace(mn_isi, mx_isi, int(delta_isi) * 1000)
hist_isi, bi = np.histogram(isi, bins=int(delta_isi) * 1000)
prob = (np.histogram(isi, bins=int(delta_isi) * 1000) / np.sum(hist_isi))[0]
# Model
lambd = 0.85
model = lambd * np.exp(-lambd * bins_isi)


plt.figure(4)
# plt.plot(bins_isi,model)
plt.hist(isi.values, bins_isi, normed=True)
plt.xlabel('time[s]')
plt.ylabel('InterSpikes/ms')
plt.ylim(0, 4)
plt.title('Tetrode:5  Neuron:2')

plt.show()

# Position

spike_pos = spike_pos.assign(x_pos=x_pos)
spike_pos = spike_pos.assign(y_pos=y_pos)

plt.figure(5)
plt.scatter(spike_pos['x_pos'], spike_pos['y_pos'], color='red', alpha=0.5)
plt.plot(x_pos, y_pos, alpha=0.5)
plt.xlabel('x_position[cm]')
plt.ylabel('y_position[cm]')
plt.title('Tetrode:5  Neuron:2')

plt.show()

# Linear Position

lin_pos = linear_position.to_frame()
plt.figure(6)
plt.scatter(spike_pos.index, spike_pos['linear_pos'], color='red', alpha=0.5)
plt.plot(lin_pos.index, lin_pos.values, alpha=0.5)
plt.xlabel('Time[s]')
plt.ylabel('Linear_position[cm]')
plt.title('Tetrode:5  Neuron:2')

plt.show()


# Rate Vs. Position

#idxx = np.array([spike_position['linear_pos']])
idxx = np.array([linear_position.values])
mxx = np.amax(idxx)
mnn = np.amin(idxx)
deltaa = mxx - mnn
binss = np.linspace(mnn, mxx, int(deltaa))

plt.figure(7)
plt.hist(spike_pos['linear_pos'], bins=int(deltaa), range=(mnn, mxx))
plt.xlabel('position[cm]')
plt.ylabel('Number of Spike')
plt.title('Tetrode:5  Neuron:2')


count = np.histogram(spike_pos['linear_pos'],
                     bins=int(deltaa), range=(mnn, mxx))[0]
occupancy = (np.histogram(linear_position.values, bins=int(
    deltaa), range=(mnn, mxx))[0]) * (1 / 1500)

rate = np.divide(count, occupancy)
pos_rate = np.histogram(rate, bins=int(deltaa), range=(mnn, mxx))


plt.figure(8)
plt.bar(binss, rate)
plt.xlabel('position[cm]')
plt.ylabel('Number of Spike/sec')
plt.title('Tetrode:5  Neuron:2')


# Rate Vs. speed

#idxxx = np.array([spike_position['speed']])
idxxx = np.array([speed.values])
mxxx = np.amax(idxxx)
mnnn = np.amin(idxxx)
deltaaa = mxxx - mnnn
binsss = np.linspace(int(mnnn), int(mxxx), int(deltaaa))
rates = np.histogram(spike_pos['speed'], binsss)[0]

plt.figure(9)
plt.hist(spike_pos['speed'], binsss)
plt.xlabel('speed')
plt.ylabel('Number of Spike')
plt.title('Tetrode:5  Neuron:2')


countt = np.histogram(spike_pos['speed'], bins=int(
    deltaaa), range=(mnnn, mxxx))[0]
occupancyy = (np.histogram(speed.values, bins=int(
    deltaaa), range=(mnnn, mxxx))[0]) * (1 / 1500)

ratee = np.divide(countt, occupancyy)
speed_rate = np.histogram(ratee, bins=int(deltaaa), range=(mnnn, mxxx))


plt.figure(10)
plt.bar(binsss, ratee)
plt.xlabel('speed')
plt.ylabel('Number of Spike/sec')
plt.title('Tetrode:5  Neuron:2')


# Rate Vs. Head Direction

#ind = np.array([spike_position['head_direction']])
ind = np.array([head_direction.values])
mxd = np.amax(ind)
mnd = np.amin(ind)
deltad = mxd - mnd
binsd = np.linspace(mnd, mxd, int(deltad), endpoint=True)
rated = np.histogram(spike_pos['head_direction'], binsd)[0]


plt.figure(11)
plt.hist(spike_pos['head_direction'], binsd)
plt.xlabel('head_direction')
plt.ylabel('Number of Spike')
plt.title('Tetrode:5  Neuron:2')


countd = np.histogram(spike_pos['head_direction'], binsd)[0]
occupancyd = (np.histogram(head_direction.values, binsd)[0]) * (1 / 1500)

rated = np.divide(countd, occupancyd)
head_rate = np.histogram(rated, bins=int(deltad), range=(mnd, mxd))


plt.figure(12)
plt.bar(binsd, rated)
plt.xlabel('head_direction')
plt.ylabel('Number of Spike/sec')
plt.title('Tetrode:5  Neuron:2')


# 2D Histogram X,Y

id_x = np.array([spike_position['x_position']])
id_y = np.array([spike_position['y_position']])
mx_x = np.amax(id_x)
mx_y = np.amax(id_y)
mn_x = np.amin(id_x)
mn_y = np.amin(id_y)
delta_x = mx_x - mn_x
delta_y = mx_y - mn_y
bin_x = np.array(np.linspace(int(mn_x), int(mx_x), int(delta_x)))
bin_y = np.array(np.linspace(int(mn_y), int(mx_y), int(delta_y)))

hist, xedges, yedges = np.histogram2d(
    spike_pos['x_pos'], spike_pos['y_pos'], bins=[bin_x, bin_y])

xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)

dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = hist.flatten()
fig = plt.figure(13)
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
plt.title('Tetrode:5  Neuron:2')
plt.xlabel('x_position')
plt.ylabel('y_position')
plt.show()


# 2D Histogram V,X

id_v = np.array([spike_position['speed']])
mx_v = np.amax(id_v)
mn_v = np.amin(id_v)
delta_v = mx_v - mn_v
bin_v = np.array(np.linspace(int(mn_v), int(mx_v), int(delta_v)))

hist_vx, xedges, vedges = np.histogram2d(
    spike_pos['x_pos'], spike_pos['speed'], bins=[bin_x, bin_v])

xpos, vpos = np.meshgrid(xedges[:-1] + 0.25, vedges[:-1] + 0.25)
xpos = xpos.flatten('F')
vpos = vpos.flatten('F')
zpos = np.zeros_like(xpos)

dx = 0.5 * np.ones_like(zpos)
dv = dx.copy()
dz = hist_vx.flatten()
fig = plt.figure(14)
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(xpos, vpos, zpos, dx, dv, dz, color='b', zsort='average')
plt.title('Tetrode:5  Neuron:2')
plt.xlabel('x_position[cm]')
plt.ylabel('v_position[cm/s]')
plt.show()


# 2D Histogram V,Y

hist_vy, yedges, vedges = np.histogram2d(
    spike_pos['y_pos'], spike_pos['speed'], bins=[bin_y, bin_v])

ypos, vpos = np.meshgrid(yedges[:-1] + 0.25, yedges[:-1] + 0.25)
ypos = xpos.flatten('F')
vpos = vpos.flatten('F')
zpos = np.zeros_like(xpos)

dy = 0.5 * np.ones_like(zpos)
dv = dy.copy()
dz = hist_vy.flatten()
fig = plt.figure(15)
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(ypos, vpos, zpos, dy, dv, dz, color='b', zsort='average')
plt.title('Tetrode:5  Neuron:2')
plt.xlabel('y_position[cm]')
plt.ylabel('v_position[cm/s]')
plt.show()


# Rate Vs. x & y
bins = [int(delta_x), int(delta_y)]

# count2d = np.histogram2d(spike_pos['x_position'],spike_pos['y_position'],
# bins=[int(delta_x),int(delta_y)],range=[ [mn_x,mx_x] ,[mn_y,mx_y]])[0]

count2d = np.histogram2d(spike_pos['x_position'], spike_pos['y_position'],
                         bins=bins, range=[[mn_x, mx_x], [mn_y, mx_y]])[0]
occupancy2d, xedge, yedge = np.histogram2d(
    spike_position['x_position'], spike_position['y_position'],
    bins=bins, range=[[mn_x, mx_x], [mn_y, mx_y]])

occupancy2d[occupancy2d == 0] = 1
# count2d = np.histogram2d(spike_pos['x_position'],spike_pos['y_position'],
# bins=[int(delta_x*0.1),int(delta_y*0.1)],range=[ [mn_x,mx_x] ,[mn_y,mx_y]])[0]
# occupancy2d,xedge,yedge = (np.histogram2d(spike_position['x_position'],spike_position['y_position'],
# bins=[int(delta_x*0.1),int(delta_y*0.1)]))

rate2d = np.divide(count2d, occupancy2d) * 1500
rate2d.max()
# np.unravel_index(count2d.argmax(), count2d.shape) to get index of arg max

xpos, ypos = np.meshgrid(xedge[:-1], yedge[:-1])
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)

dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = rate2d.flatten()
fig = plt.figure(16)
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
plt.title('Tetrode:5  Neuron:2')
plt.xlabel('x_position')
plt.ylabel('y_position')
plt.show()


# heat map

extent = [xedge[0], xedge[-1], yedge[0], yedge[-1]]
heatmap = rate2d
# Plot heatmap
"""
fig = plt.figure(30)
im = plt.imshow(heatmap, extent=extent)
plt.colorbar(im)
plt.xlabel('x[cm]')
plt.ylabel('y[cm]')
"""

f, axarr = plt.subplots(17)
im = axarr[0].imshow(heatmap, extent=extent)
f.colorbar(im, ax=axarr[0])
axarr[0].set_xlabel('y[cm]')
axarr[0].set_ylabel('x[cm]')
# axarr[0].set_title('')

"""
f, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

im = ax1.imshow(heatmap, extent=extent)
f.colorbar(im,ax=ax1)
ax1.set_xlabel('x[cm]')
ax1.set_ylabel('y[cm]')
"""

# Rate Vs. X,V

id_v = np.array([spike_position['speed']])
mx_v = np.amax(id_v)
mn_v = np.amin(id_v)
delta_v = mx_v - mn_v

"""count2d_xv = np.histogram2d(spike_pos['x_position'],spike_pos['speed'],
                         bins=[int(delta_x),int(delta_v)],range=[ [mn_x,mx_x] ,[mn_v,mx_v]])[0]
occupancy2d_xv,xedge,vedge = (np.histogram2d(spike_position['x_position'],spike_position['speed'],
                                         bins=[int(delta_x),int(delta_v)]))
"""
count2d_xv = np.histogram2d(spike_pos['x_position'], spike_pos['speed'],
                            bins=[int(delta_x), int(delta_v)], range=[[mn_x, mx_x], [mn_v, mx_v]])[0]
occupancy2d_xv, xedge, vedge = (np.histogram2d(spike_position['x_position'], spike_position['speed'],
                                               bins=[int(delta_x), int(delta_v)]))

occupancy2d_xv[occupancy2d_xv == 0] = 1
rate2d_xv = np.divide(count2d_xv, occupancy2d_xv) * 1500


xpos, vpos = np.meshgrid(xedge[:-1], vedge[:-1])
xpos = xpos.flatten('F')
vpos = vpos.flatten('F')
zpos = np.zeros_like(xpos)

dx = 0.5 * np.ones_like(zpos)
dv = dx.copy()
dz = rate2d_xv.flatten()
fig = plt.figure(18)
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(xpos, vpos, zpos, dx, dv, dz, color='b', zsort='average')
plt.title('Tetrode:5  Neuron:2')
plt.xlabel('x_position')
plt.ylabel('speed')
plt.show()

# heat map

extent_xv = [xedge[0], xedge[-1], vedge[0], vedge[-1]]
heatmap_xv = rate2d_xv
# Plot heatmap
"""
plt.figure(32)
plt.clf()
im_xv = plt.imshow(heatmap_xv, extent=extent_xv)
plt.colorbar(im_xv)
plt.xlabel('x[cm]')
plt.ylabel('v[cm/sec]')
"""

im = axarr[1].imshow(heatmap_xv, extent=extent_xv)
f.colorbar(im, ax=axarr[1])
axarr[1].set_xlabel('x[cm]')
axarr[1].set_ylabel('v[cm/sec]')
# axarr[0].set_title('')"""


"""
im = ax2.imshow(heatmap_xv, extent=extent_xv)
f.colorbar(im,ax=ax2)
ax2.set_xlabel('x[cm]')
ax2.set_ylabel('v[cm/sec]')

"""


# Rate Vs. y,V
"""
count2d_yv = np.histogram2d(spike_pos['y_position'],spike_pos['speed'],
                         bins=[int(delta_y),int(delta_v)],range=[ [mn_y,mx_y] ,[mn_v,mx_v]])[0]
occupancy2d_yv,yedge,vedge = (np.histogram2d(spike_position['y_position'],spike_position['speed'],
                                         bins=[int(delta_y),int(delta_v)]))

"""
count2d_yv = np.histogram2d(spike_pos['y_position'], spike_pos['speed'],
                            bins=[int(delta_y), int(delta_v)], range=[[mn_y, mx_y], [mn_v, mx_v]])[0]
occupancy2d_yv, yedge, vedge = (np.histogram2d(spike_position['y_position'], spike_position['speed'],
                                               bins=[int(delta_y), int(delta_v)]))

occupancy2d_yv[occupancy2d_yv == 0] = 1
rate2d_yv = np.divide(count2d_yv, occupancy2d_yv) * 1500


ypos, vpos = np.meshgrid(yedge[:-1], vedge[:-1])
ypos = ypos.flatten('F')
vpos = vpos.flatten('F')
zpos = np.zeros_like(ypos)

dy = 0.5 * np.ones_like(zpos)
dv = dy.copy()
dz = rate2d_yv.flatten()
fig = plt.figure(19)
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(ypos, vpos, zpos, dy, dv, dz, color='b', zsort='average')
plt.title('Tetrode:5  Neuron:2')
plt.xlabel('y_position')
plt.ylabel('speed')
plt.show()
# heat map

extent_yv = [yedge[0], yedge[-1], vedge[0], vedge[-1]]
heatmap_yv = rate2d_yv
# Plot heatmap
"""
plt.figure(34)
plt.clf()
im_yv = plt.imshow(heatmap_yv, extent=extent_yv)
plt.colorbar(im_yv)
plt.xlabel('y_position')
plt.ylabel('speed')

"""

im = axarr[2].imshow(heatmap_yv, extent=extent_yv)
f.colorbar(im, ax=axarr[2])
axarr[2].set_xlabel('y[cm]')
axarr[2].set_ylabel('v[cm/sec]')
# axarr[0].set_title('')"""


"""
im = ax3.imshow(heatmap_yv, extent=extent_yv)
f.colorbar(im,ax=ax3)
ax3.set_xlabel('y[cm]')
ax3.set_ylabel('v[cm/sec]')

"""

# Rate Vs. X,direction

id_h = np.array([spike_position['head_direction']])
mx_h = np.amax(id_h)
mn_h = np.amin(id_h)
delta_h = mx_h - mn_h


count2d_xh = np.histogram2d(spike_pos['x_position'], spike_pos['head_direction'],
                            bins=[int(delta_x), int(delta_h)], range=[[mn_x, mx_x], [mn_h, mx_h]])[0]
occupancy2d_xh, xedge, hedge = (np.histogram2d(spike_position['x_position'], spike_position['head_direction'],
                                               bins=[int(delta_x), int(delta_h)]))

occupancy2d_xh[occupancy2d_xh == 0] = 1
rate2d_xh = np.divide(count2d_xh, occupancy2d_xh) * 1500


xpos, hpos = np.meshgrid(xedge[:-1], hedge[:-1])
xpos = xpos.flatten('F')
hpos = hpos.flatten('F')
zpos = np.zeros_like(xpos)

dx = 0.5 * np.ones_like(zpos)
dh = dx.copy()
dz = rate2d_xh.flatten()
fig = plt.figure(26)
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(xpos, hpos, zpos, dx, dh, dz, color='b', zsort='average')
plt.title('Tetrode:5  Neuron:2')
plt.xlabel('x_position')
plt.ylabel('head_direction')
plt.show()


# Rate Vs. y,direction


count2d_yh = np.histogram2d(spike_pos['y_position'], spike_pos['head_direction'],
                            bins=[int(delta_y), int(delta_h)], range=[[mn_y, mx_y], [mn_h, mx_h]])[0]
occupancy2d_yh, yedge, hedge = (np.histogram2d(spike_position['y_position'], spike_position['head_direction'],
                                               bins=[int(delta_y), int(delta_h)]))

occupancy2d_yh[occupancy2d_yh == 0] = 1
rate2d_yh = np.divide(count2d_yh, occupancy2d_yh) * 1500


ypos, hpos = np.meshgrid(yedge[:-1], hedge[:-1])
ypos = ypos.flatten('F')
hpos = hpos.flatten('F')
zpos = np.zeros_like(ypos)

dy = 0.5 * np.ones_like(zpos)
dh = dy.copy()
dz = rate2d_yh.flatten()
fig = plt.figure(27)
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(ypos, hpos, zpos, dy, dh, dz, color='b', zsort='average')
plt.title('Tetrode:5  Neuron:2')
plt.xlabel('y_position')
plt.ylabel('head_direction')
plt.show()


# Autocorrelation

y = spike['is_spike']
N = len(spike['is_spike'])
#y = y[0:N/2]
yunbiased = y - np.mean(y)
ynorm = np.sum(y**2)
#acor = np.correlate(yunbiased, yunbiased, "same")/ynorm
acor = scis.correlate(yunbiased, yunbiased, "same")

#y = 2/(np.sqrt(N))
#acor = acor[int(len(acor)/2):]
plt.figure(20)
x_acrr = np.arange(-903771, 903770)
plt.scatter(x_acrr, acor)
# plt.stem(acor)
# plt.xlim(-100,100)
plt.ylabel('Autocorrelation')
plt.title('Tetrode:5  Neuron:2')

#plt.axhline(y=y, color='r', linestyle='-')
#plt.axhline(y=-y, color='r', linestyle='-')

plt.show()


############################# Rhythmic spiking  ###############################

"""
def raster(spiketime, **kwarg):

    ax = plt.gca() #current axes
    for trial, spike_time in enumerate(spiketime):
        #print (trial,spike_time)
        plt.vlines(spike_time, trial + .5, trial + 1.5)
    plt.ylim(.5 , len(spiketime)+0.5) # put limitation on y axis
    return ax

if __name__ == '__main__':

    spikes = spike_time.index
    fig = plt.figure(10)
    ax = raster(spikes)
    plt.title('raster plot')
    plt.xlabel('time')
    plt.ylabel('trial')
    plt.title('Tetrode:5  Neuron:2')
    fig.show()

"""
# Autocovariance

x = eeg['electric_potential']
num = len(x)
xunbiased = x - np.mean(x)

xp = x.iloc[:int(num / 20)]
xunbiasedp = xp - np.mean(xp)
xnormp = np.sum(xp**2)

acor_eeg = np.correlate(xunbiasedp, xunbiasedp, "same") / xnormp
#acor_eeg = scis.correlate(x, x, "same")
#x_acrr = np.arange(-903771,903770)
x_acrrp = np.arange(-45189, 45188)
plt.figure(21)
plt.scatter(x_acrrp, acor_eeg)
plt.xlim(-10000, 10000)
plt.ylim()
plt.title('Tetrode:5  Neuron:2')
plt.ylabel('Autocovariance')

#plt.axhline(y=y, color='r', linestyle='-')
#plt.axhline(y=-y, color='r', linestyle='-')

plt.show()

# Spectrum Estimator

plt.figure(22)
plt.plot(eeg.index, eeg.values)
plt.xlabel('Time[s]')
plt.ylabel('Voltage[V]')
# plt.xlim(4123,4124)
plt.show()

freq = 1500
dt = 1 / 1500
number = len(eeg)
T = number * dt

mean = np.mean(eeg.values)
var = np.var(eeg.values)
sd = np.std(eeg.values)

# FFT

xf = np.fft.fft(xunbiased)  # Fouried Transformation
"""
sxx = (2*dt**2/T)*(xf*np.conj(xf)) #Spectrum of x
df = (1/np.max(T))#Frequency resolutio
fNQ = 1/dt #Nyquist Freq
faxis = np.arange(0,fNQ,df)
"""
"""
plt.figure(13)
plt.plot(faxis,sxx)
#plt.xlim(0,100)
plt.xlabel('Freq[HZ]')
plt.ylabel('Power[$\mu$V^2/Hz]')
plt.title('Spectrum of EEG')
"""


# Regression
"""
response = x
t = response.index
responsep = np.array(x)
cos = np.cos(2*np.pi*60*t)
sin = np.sin(2*np.pi*60*t)
x1 = pd.DataFrame({'x1':cos})
x2 = pd.DataFrame({'x2':sin})
predictor = x1.assign(x2=x2)
predictor = np.array(predictor)

def regression (y,x):

    x = np.array(x).T
    x = sm.add_constant(x)
    results = sm.OLS(endog=y, exog=x).fit()

    return results

print (regression(responsep,predictor)).summary()

"""
############ Spectrum & Spectrogram

fs = 1500
#window = scis.tukey(256, alpha=1, sym=True)
f, Pxx_den = signal.welch(xunbiased, fs=1500, nperseg=100000)
window = scis.triang(1000, sym=True)
ff, tt, Sxx = scis.spectrogram(x - np.mean(x), fs, window=window, mode='psd')


plt.figure(23)
plt.subplot(2, 1, 1)
plt.semilogy(f, Pxx_den)
plt.xlim(0, 300)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Tetrode:5  Neuron:2')

plt.subplot(2, 1, 2)
mesh = plt.pcolormesh(tt, ff, Sxx)
plt.ylim(0, 30)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

# Theta rythm_Spectrigram
_, _, phs = scis.spectrogram(x - np.mean(x), fs, window=window, mode='phase')
_, _, mag = scis.spectrogram(
    x - np.mean(x), fs, window=window, mode='magnitude')

ff = pd.DataFrame(ff)
f_f = ff[ff.values <= 12]
f_f = f_f[f_f.values >= 8]
phase = phs[f_f.index]

f, ax = plt.subplots(24)
ax[0].plot(tt, phase[0, :])
ax[1].plot(tt, phase[1, :], 'r')
ax[2].plot(tt, phase[2, :], 'g')
ax[0].set_xlabel('time[sec]')
ax[0].set_ylabel('Phase')

_, axx = plt.subplots(25)
axx[0].plot(tt, mag[0, :])
axx[1].plot(tt, mag[1, :], 'r')
axx[2].plot(tt, mag[2, :], 'g')
axx[0].set_xlabel('time[sec]')
axx[0].set_ylabel('amplitude')


"""

#################################  Modeling ###################################


####################### Poisson
lambdd = np.linspace(0.01,5,100)
n_isi = len(isi)
log_likl = n_isi*np.log(lambdd)-lambdd*np.sum(isi.values)
#likl = np.power(lambdd,n_isi)*np.exp(np.multiply(-lambdd,np.sum(isi.values)))
log_lik = np.array(log_likl)
argmax = np.argmax(log_lik)

plt.figure(36)
plt.plot(lambdd,log_likl)
plt.xlabel('Lambda')
plt.ylabel('Log(Likelihood)')


### CDF

lambd_max = 1/np.mean(isi.values)
x_bin = np.linspace(0.001,2,18)
cdf = 1 - np.exp(-lambd_max*x_bin)
emp_cdf = np.cumsum(prob)



plt.figure(17)
plt.plot(x_bin,cdf,label='Model CDF')
plt.plot(x_bin,emp_cdf,label='Emprical CDF')
plt.ylim(0,1)
plt.legend()
plt.xlabel('Time[s]')
plt.ylabel('CDF')

### KS Plot
plt.figure(18)
plt.plot(cdf,emp_cdf)
plt.xlim(0,1)
plt.xlabel('Model CDF')
plt.ylabel('Emprical CDF')

plt.figure(19)
plt.plot(cdf,emp_cdf)
#plt.ylim(0,1)
plt.xlim(0,1)
plt.xlabel('Model CDF')
plt.ylabel('Emprical CDF')
plt.plot([0,1],[1.36/np.sqrt(n_isi),1+1.36/np.sqrt(n_isi)])
plt.plot([0,1],[-1.36/np.sqrt(n_isi),1-1.36/np.sqrt(n_isi)])
plt.title('KS Plot')

#################### Gaussian

del lambdd
del model

mu = np.mean(isi.values)
lambdd = 1/np.mean(1/isi.values-1/mu)
model_de = -(lambdd*(x_bin-mu)**2)/(2*x_bin*mu**2)
model = np.multiply(np.sqrt(lambdd/(2*np.pi*x_bin**3)) , np.exp(model_de) )

plt.figure(20)
plt.plot(x_bin,model)
plt.hist(isi,x_bin,normed=True)
plt.xlabel('time[s]')
plt.ylabel('InterSpikes/Sec')
plt.title('Tetrode:5  Neuron:2')






fig = plt.figure(41)
ax = fig.add_subplot(111, projection='3d')
#for c in range(3):
ax.scatter(spike_pos['x_position'],spike_pos['y_position'],spike_pos['speed'])



"""
