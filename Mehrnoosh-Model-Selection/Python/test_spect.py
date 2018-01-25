#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:57:48 2017

@author: Mehrnoosh
"""

import matplotlib.pyplot as plt
import numpy as np

from src.data_processing import (get_interpolated_position_dataframe,
                                 get_LFP_dataframe,
                                 get_spike_indicator_dataframe,
                                 make_epochs_dataframe, make_neuron_dataframe,
                                 make_tetrode_dataframe)
from src.parameters import ANIMALS, N_DAYS
from src.spectral.connectivity import Connectivity
from src.spectral.transforms import Multitaper

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


frequency_of_interest = 10
sampling_frequency = 1500
time_extent = (0, 1205.02733)
n_time_samples = ((time_extent[1] - time_extent[0]) * sampling_frequency) + 1
time = np.linspace(time_extent[0], time_extent[1],
                   num=n_time_samples, endpoint=True)
signals = np.array(eeg['electric_potential'])[:len(time)]

plt.figure(figsize=(15, 6))
plt.subplot(2, 2, 1)
plt.plot(time, signals)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Signal', fontweight='bold')
#plt.xlim((0.95, 1.05))
#plt.ylim((-10, 10))


plt.subplot(2, 2, 2)
m = Multitaper(signals,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=3,
               start_time=time[0])
c = Connectivity(fourier_coefficients=m.fft(),
                 frequencies=m.frequencies)
plt.plot(c.frequencies, c.power().squeeze())
