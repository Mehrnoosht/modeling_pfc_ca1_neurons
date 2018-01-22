#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:27:43 2018

@author: Mehrnoosh
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
ExpSineSquared, DotProduct,ConstantKernel)
from patsy import dmatrix
from time_rescale import TimeRescaling
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.parameters import ANIMALS
from src.parameters import N_DAYS
from loren_frank_data_processing import (make_epochs_dataframe,
                                         make_tetrode_dataframe,
                                         make_neuron_dataframe,
                                         get_spike_indicator_dataframe,
                                         get_interpolated_position_dataframe,
                                         get_LFP_dataframe)




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

linear_position = pd.DataFrame(linear_position)
x_pos = pd.DataFrame(x_pos)
y_pos = pd.DataFrame(y_pos)
eeg = get_LFP_dataframe(tetrode_key,ANIMALS )


linear_position.to_csv('linear_position.csv', sep=',')
spike.to_csv('spike.csv', sep=',')
x_pos.to_csv('x_position.csv', sep=',')
y_pos.to_csv('y_position.csv', sep=',')
head_direction.to_csv('direction.csv', sep=',')
speed.to_csv('speed.csv',sep=',')
eeg.to_csv('eeg.csv', sep=',')