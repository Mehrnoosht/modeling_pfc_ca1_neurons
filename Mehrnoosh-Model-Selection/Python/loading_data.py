#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:27:43 2018

@author: Mehrnoosh
"""
import pandas as pd
import scipy

from loren_frank_data_processing import (get_interpolated_position_dataframe,
                                         get_LFP_dataframe,
                                         get_spike_indicator_dataframe,
                                         make_epochs_dataframe,
                                         make_neuron_dataframe,
                                         make_tetrode_dataframe)
from src.parameters import ANIMALS

epoch_info = make_epochs_dataframe(ANIMALS)

epoch_key = ('HPa', 3, 2)
tetrode_info = make_tetrode_dataframe(ANIMALS).xs(epoch_key, drop_level=False)
tetrode_key = ('HPa', 3, 2, 4)

neuron_info = make_neuron_dataframe(ANIMALS).xs(epoch_key, drop_level=False)
neuron_key = ('HPa', 3,2,4,3)

spike = get_spike_indicator_dataframe(neuron_key, ANIMALS)
position_info = get_interpolated_position_dataframe(epoch_key, ANIMALS)

x_pos = position_info['x_position']
y_pos = position_info['y_position']
linear_distance = position_info['linear_distance']
x_pos = position_info['x_position']
y_pos = position_info['y_position']
speed = position_info['speed']
head_direction = position_info['head_direction']
eeg = get_LFP_dataframe(tetrode_key, ANIMALS)

linear_distance = pd.DataFrame(linear_distance)
pos_dict = {
    col_name: linear_distance[col_name].values
    for col_name in linear_distance.columns.values}
scipy.io.savemat('linear_position.mat', {'struct': pos_dict})

speed = pd.DataFrame(speed)
speed_dict = {col_name: speed[col_name].values for col_name in speed.columns.values}
scipy.io.savemat('speed.mat', {'struct': speed_dict})

eeg = pd.DataFrame(eeg)
eeg_dict = {col_name: eeg[col_name].values for col_name in eeg.columns.values}
scipy.io.savemat('eeg.mat', {'struct': eeg_dict})

spike = spike.to_frame('is_spike')
spike_dict = {
    col_name: spike[col_name].values for col_name in spike.columns.values}
scipy.io.savemat('spike.mat', {'struct': spike_dict})

time = spike.index.total_seconds()
time = pd.DataFrame(time)
time_dict = {
    col_name: time[col_name].values for col_name in time.columns.values}
scipy.io.savemat('time.mat', {'struct': time_dict})

head_direction = pd.DataFrame(head_direction)
head_direction_dict = {
    col_name: head_direction[col_name].values
    for col_name in head_direction.columns.values}
scipy.io.savemat('direction.mat', {'struct': head_direction_dict})

'''
linear_distance.to_csv('linear_distance.csv', sep=',')
spike.to_csv('spike.csv', sep=',')
x_pos.to_csv('x_position.csv', sep=',')
y_pos.to_csv('y_position.csv', sep=',')
head_direction.to_csv('direction.csv', sep=',')
speed.to_csv('speed.csv', sep=',')
eeg.to_csv('eeg.csv', sep=',')
'''
