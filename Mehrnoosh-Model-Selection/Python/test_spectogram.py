#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:33:21 2017

@author: Mehrnoosh
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import signal
import scipy.signal as scis
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
from src.parameters import ANIMALS
from src.parameters import N_DAYS
from src.data_processing import make_epochs_dataframe
from src.data_processing import make_tetrode_dataframe
from src.data_processing import make_neuron_dataframe
from src.data_processing import get_spike_indicator_dataframe
from src.data_processing import get_interpolated_position_dataframe
from src.data_processing import get_LFP_dataframe
from src.spectral.transforms import Multitaper
from src.spectral.connectivity import Connectivity


days = range(1, N_DAYS + 1)
epoch_info = make_epochs_dataframe(ANIMALS, days)
tetrode_info = make_tetrode_dataframe(ANIMALS)

epoch_key = ('HPa', 6, 2)
tetrode_key = ('HPa',6,2,5)
neuron_info = make_neuron_dataframe(ANIMALS)
neuron_key = ('HPa', 6, 2, 5, 2)

spike = get_spike_indicator_dataframe(neuron_key, ANIMALS)
linear_position = get_interpolated_position_dataframe(epoch_key, ANIMALS)['linear_position']
x_pos =  get_interpolated_position_dataframe(epoch_key, ANIMALS)['x_position']
y_pos = get_interpolated_position_dataframe(epoch_key, ANIMALS)['y_position']
speed = get_interpolated_position_dataframe(epoch_key, ANIMALS)['speed']
head_direction = get_interpolated_position_dataframe(epoch_key, ANIMALS)['head_direction']

spike_position =  spike.assign(linear_pos = linear_position, x_position = x_pos,y_position = y_pos,speed=speed,head_direction=head_direction)
spike_pos = spike_position[spike['is_spike']==1]


eeg = get_LFP_dataframe(tetrode_key,ANIMALS )


frequency_of_interest = 8
sampling_frequency = 1500
time_extent = (0, 1205.02733)
n_time_samples = ((time_extent[1] - time_extent[0]) * sampling_frequency) + 1
time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)
signals = np.array(eeg['electric_potential'])[:len(time)]



m = Multitaper(signals,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=8,
               time_window_duration=0.6,
               time_window_step=0.500,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies,
                 time=m.time)


mesh = plt.pcolormesh(c.time, c.frequencies, c.power().squeeze().T,
                             vmin=0.0, vmax=0.03, cmap='viridis')
#axes[1, 0].set_ylim((0, 300))
plt.axvline(time[int(np.fix(n_time_samples / 2))], color='black')


#plt.tight_layout()
cb = plt.colorbar(mesh, orientation='horizontal',
                  shrink=.5, aspect=15, pad=0.1, label='Power')
cb.outline.set_linewidth(0)

'''
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 9))
axes[0, 0].plot(time, signals)
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].set_title('Signal', fontweight='bold')
#axes[0, 0].set_xlim((24.90, 25.10))
#axes[0, 0].set_ylim((-10, 10))


m = Multitaper(signals,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=3,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies)
axes[0, 1].plot(c.frequencies, c.power().squeeze())



m = Multitaper(signals,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=3,
               time_window_duration=0.6,
               time_window_step=0.500,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies,
                 time=m.time)
mesh = axes[1, 0].pcolormesh(c.time, c.frequencies, c.power().squeeze().T,
                             vmin=0.0, vmax=0.03, cmap='viridis')
#axes[1, 0].set_ylim((0, 300))
axes[1, 0].axvline(time[int(np.fix(n_time_samples / 2))], color='black')


plt.tight_layout()
cb = fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation='horizontal',
                  shrink=.5, aspect=15, pad=0.1, label='Power')
cb.outline.set_linewidth(0)

'''

