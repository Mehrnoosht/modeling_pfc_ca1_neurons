#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 17:37:33 2017

@author: Mehrnoosh
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
from patsy import dmatrix
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel, DotProduct,
                                              ExpSineSquared, Matern,
                                              RationalQuadratic)

from src.data_processing import (get_interpolated_position_dataframe,
                                 get_spike_indicator_dataframe,
                                 make_epochs_dataframe, make_neuron_dataframe,
                                 make_tetrode_dataframe)
from src.parameters import ANIMALS, N_DAYS
from time_rescale import TimeRescaling

########################## Loading Data #######################################
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

spike_position = spike.assign(linear_pos=linear_position,
                              x_position=x_pos, y_position=y_pos, speed=speed,
                              head_direction=head_direction)

linear_position = pd.DataFrame(linear_position)
x_pos = pd.DataFrame(x_pos)
y_pos = pd.DataFrame(y_pos)

spike_pos = spike_position[spike['is_spike'] == 1]


a_dict = {
    col_name: linear_position[col_name].values
    for col_name in linear_position.columns.values}

# optional if you want to save the index as an array as well:
# a_dict[df.index.name] = df.index.values
scipy.io.savemat('linear_position.mat', {'struct': a_dict})

linear_position.to_csv('linear_position.csv', sep=',')
spike.to_csv('spike.csv', sep=',')
x_pos.to_csv('x_position.csv', sep=',')
y_pos.to_csv('y_position.csv', sep=',')
head_direction.to_csv('direction.csv', sep=',')
speed.to_csv('speed.csv', sep=',')

################################## Mode 0 #####################################

y = spike_position['is_spike']

ones_column = np.ones(y.shape)
design_matrix = pd.DataFrame(ones_column, columns=['one'])


y = y.values

# plt.scatter(design_matrix,func_cov_mode0)


######################## linear_position is the Covariate #####################

# True Model
idxx = np.array([spike_position['linear_pos'].values])
mxx = np.amax(idxx)
mnn = np.amin(idxx)
deltaa = mxx - mnn
binss = np.linspace(mnn, mxx, int(deltaa))
count, bc = np.histogram(spike_pos['linear_pos'], bins=int(deltaa))
occupancy, bo = np.histogram(
    spike_position['linear_pos'].values, bins=int(deltaa))
rate = np.divide(count, occupancy)
[spike_binss, pos_bin] = np.histogram(
    spike_pos['linear_pos'], 375, range=(mnn, mxx))

plt.figure(2)
rescaled = TimeRescaling(rate, spike_binss)

fig, axes = plt.subplots(1, 1, figsize=(12, 6))
rescaled.plot_ks(ax=axes)


# Spline Model


c_pt = np.array([-190, -150, -100, -50, 0, 60, 80, 120, 150, 200])
s = 0.5
pos_spike = spike_position['linear_pos'].values
X = pd.DataFrame({'cpt1': [], 'cpt2': [], 'cpt3': [], 'cpt4': [], 'cpt5': [],
                  'cpt6': [], 'cpt8': [], 'cpt9': [], 'cpt10': []})
spline = np.zeros([len(spike_position.index), 10])
spline = pd.DataFrame(spline)

num_c_pts = len(c_pt)
rang = np.linspace(-200, 200)

for i in np.arange((len(spike_position.index))):
    nearest_c_pt_index = np.max(np.where(c_pt < pos_spike[i]))
    nearest_c_pt_time = c_pt[nearest_c_pt_index]
    next_c_pt_time = c_pt[nearest_c_pt_index + 1]
    u = (i - nearest_c_pt_time) / (next_c_pt_time - nearest_c_pt_time)
    p = np.matmul(np.array([u**3, u**2, u, 1]),
                  np.array([[-s, 2 - s, s - 2, s],
                            [2 * s, s - 3, 3 - 2 * s, -s],
                            [-s, 0, s, 0], [0, 1, 0, 0]]))
    if nearest_c_pt_index < 2:
        nearest_c_pt_index = 2

    concat = np.concatenate((np.zeros(nearest_c_pt_index - 2), p))
    spline.iloc[i] = np.concatenate(
        (concat, np.zeros(num_c_pts - 4 - (nearest_c_pt_index - 2))))

spline.columns = ['cpt1', 'cpt2', 'cpt3', 'cpt4',
                  'cpt5', 'cpt6', 'cpt7', 'cpt8', 'cpt9', 'cpt10']
a_dict = {
    col_name: spline[col_name].values for col_name in spline.columns.values}

# optional if you want to save the index as an array as well:
# a_dict[df.index.name] = df.index.values
scipy.io.savemat('spline.mat', {'struct': a_dict})

spline.to_csv('spline.csv', sep=',')

glm_spline = sm.GLM(y, spline, family=sm.families.Poisson())
glm_poisson_spl2d = glm_spline.fit()
print(glm_poisson_spl2d.summary())
beta_spl2d = glm_poisson_spl2d.params
lambda_spl2d = glm_poisson_spl2d.mu
