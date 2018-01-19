#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 12:50:31 2017

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
                                         get_interpolated_position_dataframe)




########################## Loading Data #######################################
epoch_info = make_epochs_dataframe(ANIMALS)
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

spike_position =  spike.assign(linear_pos = linear_position,
                               x_position = x_pos,y_position = y_pos,speed=speed,
                               head_direction=head_direction)
spike_pos = spike_position[spike['is_spike']==1]


################################## Mode 0 #####################################
 
y =  spike_position['is_spike']

ones_column = np.ones(y.shape)
design_matrix = pd.DataFrame(ones_column,columns=['one'])



y = y.values

glm_mode0 = sm.GLM(y ,design_matrix.values, family=sm.families.Poisson())
glm_poisson_mode0 =  glm_mode0.fit()
print(glm_poisson_mode0.summary())
beta_mode0 = glm_poisson_mode0.params
lambda_mode0 = np.exp(np.sum(np.multiply(beta_mode0,design_matrix.values),axis=1))

#plt.scatter(design_matrix,func_cov_mode0)


######################## linear_position is the Covariate #####################
 
 
#### Model Fit1
design_matrix_l = design_matrix.assign(linear_pos=spike_position['linear_pos'].values)
glm_l = sm.GLM(y ,design_matrix_l.values, family=sm.families.Poisson())
glm_poisson_l=  glm_l.fit()
print(glm_poisson_l.summary())
beta_l = glm_poisson_l.params
lambda_l = glm_poisson_l.mu

#### Model Fit2
design_matrix_ll = design_matrix_l.assign(linear_pos_pw2=np.power(spike_position['linear_pos'].values,2))
glm_ll = sm.GLM(y ,design_matrix_ll.values, family=sm.families.Poisson())
glm_poisson_ll =  glm_ll.fit()
print(glm_poisson_ll.summary())
beta_ll = glm_poisson_ll.params
lambda_ll = glm_poisson_ll.mu

#### Model Fit3 
dsgn = design_matrix
mu  = np.linspace(np.min(spike_position['linear_pos'].values),np.max(spike_position['linear_pos'].values),10)
variance = 50

def gaussian_rbf(x, mean, variance):
    rbf_gaussian = np.exp(-((x - mean) ** 2) / (2 * variance))
    return rbf_gaussian
 
rbf = []
for n in np.arange(len(mu)) :

    rbf.append( gaussian_rbf(spike_position['linear_pos'].values,mu[n],100))


design_matrix_lrbf = dsgn.assign(rbf0 = rbf[0],rbf1=rbf[1],rbf2=rbf[2],rbf3=rbf[3],rbf4=rbf[4],
                                rbf5 = rbf[5],rbf6 = rbf[6],rbf7=rbf[7],rbf8=rbf[8]
                                ,rbf9=rbf[9])
#plt.plot(spike_position['linear_pos'].values,rbf[4])
#plt.figure(); [plt.scatter(spike_position['linear_pos'], r) for r in rbf];
glm_lrbf = sm.GLM(y ,design_matrix_lrbf.values, family=sm.families.Poisson())
glm_poisson_lrbf =  glm_lrbf.fit()
print(glm_poisson_lrbf.summary())
beta_lrbf = glm_poisson_lrbf.params
lambda_lrbf = glm_poisson_lrbf.mu

#### Model fit 4
design_matrixspline = dmatrix('bs(cov, df=8) -1', dict(cov = spike_position['linear_pos'].values))
glm_spline = sm.GLM(y, design_matrixspline,family=sm.families.Poisson())
glm_poisson_spl =  glm_spline.fit()
print(glm_poisson_spl.summary())
beta_spl = glm_poisson_spl.params
lambda_spl = glm_poisson_spl.mu 

##### Model fit 5

design_matrixspline1 = dmatrix("cc(cov, df=6)", dict(cov = spike_position['linear_pos'].values))
glm_spline1 = sm.GLM(y, design_matrixspline1,family=sm.families.Poisson())
glm_poisson_spl1 =  glm_spline1.fit()
print(glm_poisson_spl1.summary())
beta_spl1 = glm_poisson_spl1.params
lambda_spl1 = glm_poisson_spl1.mu 


##### True Model
idxx = np.array([spike_position['linear_pos'].values])
mxx = np.amax(idxx)
mnn = np.amin(idxx)
deltaa = mxx-mnn
binss =  np.linspace(mnn,mxx,int(deltaa))
count = np.histogram(spike_pos['linear_pos'],bins=int(deltaa),range=(mnn,mxx))[0]
occupancy = (np.histogram(spike_position['linear_pos'].values,bins=int(deltaa),range=(mnn,mxx))[0])*(1/1500)
rate = np.divide(count,occupancy)
pos_rate = np.histogram(rate,bins=int(deltaa),range=(mnn,mxx))

################# Evaluation

#1. Visualization
plt.figure(1)
plt.bar(binss,rate)
plt.plot(spike_position['linear_pos'],lambda_lrbf*1500,color='yellow')
plt.plot(spike_position['linear_pos'],lambda_l*1500,color='red')
plt.plot(spike_position['linear_pos'],lambda_ll*1500,color='green')
plt.plot(spike_position['linear_pos'],lambda_spl*1500,color='black')
plt.plot(spike_position['linear_pos'],lambda_spl*1500,color='purple')
plt.xlabel('linear_pos[cm]')
plt.ylabel('Number of Spike/sec')
plt.title('Tetrode:5  Neuron:2')

#2. Deviance 
dev_model1 = glm_poisson_l.deviance
dev_model2 = glm_poisson_ll.deviance
dev_model3 = glm_poisson_lrbf.deviance

#3. KS plot for time Rescaling ISIs

N = len(spike_pos)
lambda_l = pd.DataFrame(lambda_l, index = spike.index)
lambda_ll = pd.DataFrame(lambda_ll, index = spike.index)
lambda_lrbf = pd.DataFrame(lambda_lrbf, index = spike.index)
lambda_spl1 = pd.DataFrame(lambda_spl1, index = spike.index)
Z = np.zeros([N,1])
Z[0] = np.sum(lambda_l[lambda_l.index[0]:spike_pos.index[0]].values)
for i in np.arange(1,N):
    
    Z[i] = np.sum(lambda_l[spike_pos.index[i-1]:spike_pos.index[i]].values)

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
plt.figure(2)
plt.plot(mcdf,eCDF)
plt.plot([0,1],[1.36/np.sqrt(N),1+1.36/np.sqrt(N)])
plt.plot([0,1],[-1.36/np.sqrt(N),1-1.36/np.sqrt(N)])
plt.xlabel('Model CDF')
plt.ylabel('Emprical CDF')
plt.title('KS Plot For linear_pos')


Z = np.zeros([N,1])
Z[0] = np.sum(lambda_ll[lambda_ll.index[0]:spike_pos.index[0]].values)
for i in np.arange(1,N):
    
    Z[i] = np.sum(lambda_ll[spike_pos.index[i-1]:spike_pos.index[i]].values)

[zval,eCDF] = ecdf(Z.reshape(-1))
mcdf = 1-np.exp(-zval)
plt.figure(3)
plt.plot(mcdf,eCDF)
plt.plot([0,1],[1.36/np.sqrt(N),1+1.36/np.sqrt(N)])
plt.plot([0,1],[-1.36/np.sqrt(N),1-1.36/np.sqrt(N)])
plt.xlabel('Model CDF')
plt.ylabel('Emprical CDF')
plt.title('KS Plot For linear_pos')

Z = np.zeros([N,1])
Z[0] = np.sum(lambda_lrbf[lambda_lrbf.index[0]:spike_pos.index[0]].values)
for i in np.arange(1,N):
    
    Z[i] = np.sum(lambda_lrbf[spike_pos.index[i-1]:spike_pos.index[i]].values)

[zval,eCDF] = ecdf(Z.reshape(-1))
mcdf = 1-np.exp(-zval)
plt.figure(4)
plt.plot(mcdf,eCDF)
plt.plot([0,1],[1.36/np.sqrt(N),1+1.36/np.sqrt(N)])
plt.plot([0,1],[-1.36/np.sqrt(N),1-1.36/np.sqrt(N)])
plt.xlabel('Model CDF')
plt.ylabel('Emprical CDF')
plt.title('KS Plot For Gaussian RBF')

Z = np.zeros([N,1])
Z[0] = np.sum(lambda_spl1[lambda_spl1.index[0]:spike_pos.index[0]].values)
for i in np.arange(1,N):
    
    Z[i] = np.sum(lambda_spl1[spike_pos.index[i-1]:spike_pos.index[i]].values)

[zval,eCDF] = ecdf(Z.reshape(-1))
mcdf = 1-np.exp(-zval)
plt.figure(5)
plt.plot(mcdf,eCDF)
plt.plot([0,1],[1.36/np.sqrt(N),1+1.36/np.sqrt(N)])
plt.plot([0,1],[-1.36/np.sqrt(N),1-1.36/np.sqrt(N)])
plt.xlabel('Model CDF')
plt.ylabel('Emprical CDF')
plt.title('KS Plot For Spline')



rescaled = TimeRescaling(glm_poisson_spl1.mu,y)

fig, axes = plt.subplots(1, 1, figsize=(12, 6))
rescaled.plot_ks(ax=axes)



"""
############################ X_position is the Covariate ######################
 
#### Model Fit1
design_matrix_x = design_matrix.assign(x_position=spike_position['x_position'].values)
glm_x = sm.GLM(y ,design_matrix_x.values, family=sm.families.Poisson())
glm_poisson_x =  glm_x.fit()
print(glm_poisson_x.summary())
beta_x = glm_poisson_x.params
lambda_x = np.exp(np.sum(np.multiply(beta_x,design_matrix_x.values),axis=1))

#### Model Fit2
design_matrix_xx = design_matrix_x.assign(x_position_pw2=np.power(spike_position['x_position'].values,2))
glm_xx = sm.GLM(y ,design_matrix_xx.values, family=sm.families.Poisson())
glm_poisson_xx =  glm_xx.fit()
print(glm_poisson_xx.summary())
beta_xx = glm_poisson_xx.params
lambda_xx = np.exp(np.sum(np.multiply(beta_xx,design_matrix_xx.values),axis=1))

#### Model Fit3 
dsgn = design_matrix
mu = [40,60,70,80,90,100,110,120,130]
variance = 10
rbf = []
for n in np.arange(len(mu)) :

    rbf.append( np.exp(-(spike_position['x_position'].values-mu[n])**2/variance ))
   
design_matrix_xrbf = dsgn.assign(rbf0 = rbf[0],rbf1=rbf[1],rbf2=rbf[2],rbf3=rbf[3],rbf4=rbf[4],
                                rbf5 = rbf[5],rbf6=rbf[6],rbf7=rbf[7],rbf8=rbf[8])
glm_xrbf = sm.GLM(y ,design_matrix_xrbf.values, family=sm.families.Poisson())
glm_poisson_xrbf =  glm_xrbf.fit()
print(glm_poisson_xrbf.summary())
beta_xrbf = glm_poisson_xrbf.params
lambda_xrbf = np.exp(np.sum(np.multiply(beta_xrbf,design_matrix_xrbf.values),axis=1))


##### True Model
idxx = np.array([spike_position['x_position'].values])
mxx = np.amax(idxx)
mnn = np.amin(idxx)
deltaa = mxx-mnn
binss =  np.linspace(mnn,mxx,int(deltaa))
count = np.histogram(spike_pos['x_position'],bins=int(deltaa),range=(mnn,mxx))[0]
occupancy = (np.histogram(spike_position['x_position'].values,bins=int(deltaa),range=(mnn,mxx))[0])*(1/1500)
rate = np.divide(count,occupancy)
pos_rate = np.histogram(rate,bins=int(deltaa),range=(mnn,mxx))

################# Evaluation

#1. Visualization
plt.figure(1)
plt.bar(binss,rate)
plt.plot(spike_position['x_position'],lambda_x*1500,color='red')
plt.plot(spike_position['x_position'],lambda_xx*1500,color='green')
plt.plot(spike_position['x_position'],lambda_xrbf*1500,color='yellow')
plt.xlabel('x_position[cm]')
plt.ylabel('Number of Spike/sec')
plt.title('Tetrode:5  Neuron:2')

#2. Deviance 
dev_model1 = glm_poisson_x.deviance
dev_model2 = glm_poisson_xx.deviance
dev_model3 = glm_poisson_xrbf.deviance

#3. KS plot for time Rescaling ISIs

N = len(spike_pos)
lambda_x = pd.DataFrame(lambda_x, index = spike.index)
lambda_xx = pd.DataFrame(lambda_xx, index = spike.index)
lambda_xrbf = pd.DataFrame(lambda_xrbf, index = spike.index)
Z = np.zeros([N,1])
Z[0] = np.sum(lambda_x[lambda_x.index[0]:spike_pos.index[0]].values)
for i in np.arange(1,N):
    
    Z[i] = np.sum(lambda_x[spike_pos.index[i-1]:spike_pos.index[i]].values)

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
plt.figure(2)
plt.plot(mcdf,eCDF)
plt.xlabel('Model CDF')
plt.ylabel('Emprical CDF')
plt.title('KS Plot For x_position')


Z = np.zeros([N,1])
Z[0] = np.sum(lambda_xx[lambda_xx.index[0]:spike_pos.index[0]].values)
for i in np.arange(1,N):
    
    Z[i] = np.sum(lambda_xx[spike_pos.index[i-1]:spike_pos.index[i]].values)

[zval,eCDF] = ecdf(Z.reshape(-1))
mcdf = 1-np.exp(-zval)
plt.figure(3)
plt.plot(mcdf,eCDF)
plt.xlabel('Model CDF')
plt.ylabel('Emprical CDF')
plt.title('KS Plot For x_position')

Z = np.zeros([N,1])
Z[0] = np.sum(lambda_xrbf[lambda_xrbf.index[0]:spike_pos.index[0]].values)
for i in np.arange(1,N):
    
    Z[i] = np.sum(lambda_xrbf[spike_pos.index[i-1]:spike_pos.index[i]].values)

[zval,eCDF] = ecdf(Z.reshape(-1))
mcdf = 1-np.exp(-zval)
plt.figure(4)
plt.plot(mcdf,eCDF)
plt.xlabel('Model CDF')
plt.ylabel('Emprical CDF')
plt.title('KS Plot For Gaussian RBF')
"""
 
############################ X and Y are the Covariates #######################
 

########## Model fit 1
design_matrix_mode1 = design_matrix.assign(x_position=spike_position['x_position'].values,
                                     y_position = spike_position['y_position'].values)



glm_mode1 = sm.GLM(y ,design_matrix_mode1.values, family=sm.families.Poisson())
glm_poisson_mode1 =  glm_mode1.fit()
print(glm_poisson_mode1.summary())
beta_mode1 = glm_poisson_mode1.params
lambda_mode1 = np.exp(np.sum(np.multiply(beta_mode1,design_matrix_mode1.values),axis=1))

func_cov_mode1 = np.sum(np.multiply(beta_mode1[1:3],
                                    design_matrix_mode1.values[:,[1,2]]),axis=1)


################# Evaluation

x_new = np.linspace(-500, 500, 100)
y_new = np.linspace(-500,500,100)
xv, yv = np.meshgrid(x_new, y_new)
xvv = beta_mode1[1]*xv
yvv = beta_mode1[2]*yv

lambda_mode = np.exp(beta_mode1[0]+xvv+yvv)

fig = plt.figure(6)
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(xv, yv, lambda_mode, rstride=10, cstride=10)


###KS Plot
 
N = len(spike_pos)
lambda_mode1 = pd.DataFrame(lambda_mode1, index = spike.index)
Z1 = np.zeros([N,1])
Z1[0] = np.sum(lambda_mode1[lambda_mode1.index[0]:spike_pos.index[0]].values)
for i in np.arange(1,N):
    
    Z1[i] = np.sum(lambda_mode1[spike_pos.index[i-1]:spike_pos.index[i]].values)


[zval1,eCDF1] = ecdf(Z1.reshape(-1))
mcdf1 = 1-np.exp(-zval1)
plt.figure(7)
plt.plot(mcdf1,eCDF1)
plt.plot([0,1],[1.36/np.sqrt(N),1+1.36/np.sqrt(N)])
plt.plot([0,1],[-1.36/np.sqrt(N),1-1.36/np.sqrt(N)])
plt.xlabel('Model CDF')
plt.ylabel('Emprical CDF')
plt.title('KS Plot For X and Y')

    

############## Model fit 2 : X,Y,X^2,y^2 
xpw2 = np.power(spike_position['x_position'].values,2)
ypw2 = np.power(spike_position['y_position'].values,2)

design_matrix_mode2 = design_matrix_mode1.assign(x_position_pw2=xpw2,
                                     y_position_pw2 = ypw2)



design_matrix_mode2 = design_matrix_mode2.values


glm_mode2 = sm.GLM(y ,design_matrix_mode2, family=sm.families.Poisson())
glm_poisson_mode2 =  glm_mode2.fit()
print(glm_poisson_mode2.summary())
beta_mode2 = glm_poisson_mode2.params
lambda_mode2 = np.exp(np.sum(np.multiply(beta_mode2,design_matrix_mode2),axis=1))

func_cov_mode2 = np.sum(np.multiply(beta_mode2[1:3],
                                    design_matrix_mode2[:,[1,2]]),axis=1)


################################# Evaluation
x_new = np.linspace(40, 160, 100)
y_new = np.linspace(0,120,100)
xv, yv = np.meshgrid(x_new, y_new)
xvv = beta_mode2[1]*xv
yvv = beta_mode2[2]*yv
x2 = beta_mode2[3]*xv**2
y2 = beta_mode2[4]*yv**2
lambda_mode = np.exp(beta_mode2[0]+xvv+yvv+x2+y2)

fig = plt.figure(8)
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(xv, yv, lambda_mode, rstride=10, cstride=10)


##KS Plot

lambda_mode2 = pd.DataFrame(lambda_mode2, index = spike.index)
Z2 = np.zeros([N,1])
Z2[0] = np.sum(lambda_mode2[lambda_mode2.index[0]:spike_pos.index[0]].values)
for i in np.arange(1,N):
    
    Z2[i] = np.sum(lambda_mode2[spike_pos.index[i-1]:spike_pos.index[i]].values)


[zval2,eCDF2] = ecdf(Z2.reshape(-1))
mcdf2 = 1-np.exp(-zval2)
plt.figure(9)
plt.plot(mcdf2,eCDF2)
plt.plot([0,1],[1.36/np.sqrt(N),1+1.36/np.sqrt(N)])
plt.plot([0,1],[-1.36/np.sqrt(N),1-1.36/np.sqrt(N)])
plt.xlabel('Model CDF')
plt.ylabel('Emprical CDF')
plt.title('KS Plot For X,Y,X^2,Y^2')

################ Model Fit3: Gaussian Radial Base Function 
dsgn = design_matrix
xgrid = [64,65,65,65,69,72,80,85,86,88,93,102,103,108,109,100,93,98,101,113,130]
ygrid = [64,52,42,38,24,22,23,20,93,98,98,100,104,95,86,45,22,22,23,26,28]
mu = np.c_[xgrid,ygrid]
variance = 4

data = np.c_[spike_position['x_position'].values , spike_position['y_position'].values ]
ds = pd.DataFrame({'A' : []})
rbf =[]
for n in np.arange(len(mu)) :

    rbf.append( np.exp(-np.sum((data-mu[n])**2,1)/variance**2 ))
    
    
design_matrix_rbf = dsgn.assign(rbf0 = rbf[0],rbf1=rbf[1],rbf2=rbf[2],rbf3=rbf[3],rbf4=rbf[4],
                                rbf5 = rbf[5],rbf6 = rbf[6],rbf7=rbf[7],rbf8=rbf[8]
                                ,rbf9=rbf[9],rbf10=rbf[10],rbf11 = rbf[11],rbf12=rbf[12],rbf13=rbf[13],
                                rbf14=rbf[14],rbf15=rbf[15],rbf16 = rbf[16],rbf17=rbf[17],rbf18=rbf[18],
                                rbf19=rbf[19],rbf20=rbf[20])

glm_rbf = sm.GLM(y ,design_matrix_rbf.values, family=sm.families.Poisson())
glm_poisson_rbf =  glm_rbf.fit()
print(glm_poisson_rbf.summary())
beta_rbf = glm_poisson_rbf.params
lambda_rbf = np.exp(np.sum(np.multiply(beta_rbf,design_matrix_rbf.values),axis=1))
# or lambda_rbf = gm_poisson_rbf.mu

################# Evaluation
x_new = np.linspace(40, 160, 100)
y_new = np.linspace(0,120,100)
xv, yv = np.meshgrid(x_new, y_new)
data_new = np.c_[xv.ravel(),yv.ravel()]
rbf_new =[]
for n in np.arange(len(mu)) :

    rbf_new.append( np.exp(-np.sum((data_new-mu[n])**2,1)/variance**2 ))
    
    
lambda_mode_rbf = np.exp(beta_rbf[0]+beta_rbf[1]*rbf_new[0]+beta_rbf[2]*rbf_new[1]+beta_rbf[3]*rbf_new[2]
                        +beta_rbf[4]*rbf_new[3]+beta_rbf[5]*rbf_new[4]+beta_rbf[6]*rbf_new[5]+beta_rbf[7]
                        *rbf_new[6]+beta_rbf[8]*rbf_new[7]+beta_rbf[9]*rbf_new[8]
                        +beta_rbf[10]*rbf_new[9]+beta_rbf[11]*rbf_new[10]+beta_rbf[12]*rbf_new[11]+
                        beta_rbf[13]*rbf_new[12]+beta_rbf[14]*rbf_new[13]+beta_rbf[15]*rbf_new[14]
                        +beta_rbf[16]*rbf_new[15]+beta_rbf[17]*rbf_new[16]+beta_rbf[18]*rbf_new[17]+
                        beta_rbf[19]*rbf_new[18]+beta_rbf[20]*rbf_new[19]+beta_rbf[21]*rbf_new[20]
                        )

fig = plt.figure(10)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xv, yv, lambda_mode_rbf.reshape(xv.shape), rstride=10, cstride=10)


"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = itertools.cycle(["r", "b", "g"])
ax.scatter(design_matrix_mode1.values[...,1], design_matrix_mode1.values[...,2], func_cov_mode1, zdir='z',
          s=20, c=next(colors), depthshade=True)

 """ 
###KS Plot
 
rescaled = TimeRescaling(lambda_rbf,y)

fig, axes = plt.subplots(1, 1, figsize=(12, 6))
rescaled.plot_ks(ax=axes)

#
 
N = len(spike_pos)
lambda_rbf = pd.DataFrame(lambda_rbf, index = spike.index)
Z1 = np.zeros([N,1])
Z1[0] = np.sum(lambda_rbf[lambda_rbf.index[0]:spike_pos.index[0]].values)
for i in np.arange(1,N):
    
    Z1[i] = np.sum(lambda_rbf[spike_pos.index[i-1]:spike_pos.index[i]].values)


[zval1,eCDF1] = ecdf(Z1.reshape(-1))
mcdf1 = 1-np.exp(-zval1)
plt.figure(11)
plt.plot(mcdf1,eCDF1)
plt.plot([0,1],[1.36/np.sqrt(N),1+1.36/np.sqrt(N)])
plt.plot([0,1],[-1.36/np.sqrt(N),1-1.36/np.sqrt(N)])
plt.xlabel('Model CDF')
plt.ylabel('Emprical CDF')
plt.title('KS Plot For X and Y')

    
################# Spline 

design_matrixspline_2d = dmatrix("te(cr(x, df=4), cc(y, df=4))",
                                 dict(x = spike_position['x_position'].values,y = spike_position['y_position'].values ))
glm_spline_2d = sm.GLM(y, design_matrixspline_2d,family=sm.families.Poisson())
glm_poisson_spl2d =  glm_spline_2d.fit()
print(glm_poisson_spl2d.summary())
beta_spl2d = glm_poisson_spl2d.params
lambda_spl2d = glm_poisson_spl2d.mu 

################# Evaluation

 
###KS Plot
 
rescaled = TimeRescaling(lambda_spl2d,y)

fig, axes = plt.subplots(1, 1, figsize=(12, 6))
rescaled.plot_ks(ax=axes)

#
 
N = len(spike_pos)
lambda_rbf = pd.DataFrame(lambda_rbf, index = spike.index)
Z1 = np.zeros([N,1])
Z1[0] = np.sum(lambda_rbf[lambda_rbf.index[0]:spike_pos.index[0]].values)
for i in np.arange(1,N):
    
    Z1[i] = np.sum(lambda_rbf[spike_pos.index[i-1]:spike_pos.index[i]].values)


[zval1,eCDF1] = ecdf(Z1.reshape(-1))
mcdf1 = 1-np.exp(-zval1)
plt.figure(11)
plt.plot(mcdf1,eCDF1)
plt.plot([0,1],[1.36/np.sqrt(N),1+1.36/np.sqrt(N)])
plt.plot([0,1],[-1.36/np.sqrt(N),1-1.36/np.sqrt(N)])
plt.xlabel('Model CDF')
plt.ylabel('Emprical CDF')
plt.title('KS Plot For X and Y')







