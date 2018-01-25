#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 09:34:11 2017

@author: Mehrnoosh
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel, DotProduct,
                                              ExpSineSquared, Matern,
                                              RationalQuadratic)

x = np.linspace(-10, 10)[:, np.newaxis]
mean = np.atleast_2d([7.5, -5.0])
std_dev = 1.0
kernel = RBF(length_scale=[std_dev, std_dev])
x1_grid, x2_grid = np.meshgrid(x, x)
data = np.c_[x1_grid.ravel(), x2_grid.ravel()]
Z = kernel(mean, data)
plt.contour(x1_grid, x2_grid, Z.reshape(x1_grid.shape), 20, cmap='RdGy')

x, y, z, d = np.random.rand(4, 50)
rbfi = Rbf(x, y, z, d)  # radial basis function interpolator instance
xi = yi = zi = np.linspace(0, 1, 20)
di = rbfi(xi, yi, zi)   # interpolated values

plt.figure(8)
plt.plot(xi, k)
