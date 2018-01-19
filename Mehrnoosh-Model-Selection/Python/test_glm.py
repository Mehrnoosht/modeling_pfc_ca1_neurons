#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 20:12:53 2017

@author: Mehrnoosh
"""
import numpy as np
import pandas as pd
import scipy.io as load
import statsmodels.api as sm
import statsmodels.distributions as smf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = load.loadmat('glm_data.mat')

y = pd.DataFrame(data['spikes_binned'],columns=['spike count'])
ones_column = np.ones(y.shape)
design_matrix = pd.DataFrame(ones_column,columns=['one'])
design_matrix = design_matrix.assign(x_position=data['xN'],y_position=data['yN'])

glm = sm.GLM(y.values ,design_matrix.values, family=sm.families.Poisson())
glm_poisson =  glm.fit()
beta = glm_poisson.params
lambd = np.exp(np.sum(np.multiply(beta,design_matrix.values),axis=1))

############################# Evaluation Model

## KS Plot


spike = y[y['spike count']>0]
N = len(spike)
Z = np.zeros(spike.shape)


Z[1] = np.sum(lambd[0:spike.index[0]])

for i in np.arange(1,N):
    
    Z[i] = np.sum(lambd[spike.index[i-1]:spike.index[i]])

def ecdf(sample):
    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size
    return quantiles, cumprob

[zval,eCDF] = ecdf(Z.reshape(-1))
mcdf = 1-np.exp(-zval)
plt.figure(4)
plt.plot(mcdf,eCDF)
plt.xlabel('Model CDF')
plt.ylabel('Emprical CDF')
plt.title('KS Plot')

x_new = np.linspace(-1, 1, 3)
y_new = np.linspace(-1,1,3)
xv, yv = np.meshgrid(x_new, y_new)
xvv = beta[1]*xv
yvv = beta[2]*yv

lambda_mode = np.exp(beta[0]+xvv+yvv)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xv, yv, lambda_mode)

