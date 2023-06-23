#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:07:41 2023

@author: manupc
"""

import numpy as np
import matplotlib.pyplot as plt



# Convolución 2D
def conv2D(X, Y, padding=0, stride=1):

    Y_height, Y_width = Y.shape
    X_height, X_width = X.shape
    S_height = ((X_height - Y_height)+1+2*padding)//stride #// stride + 1
    S_width = ((X_width - Y_width)+1+2*padding)//stride #// stride + 1

    if padding>0:
        Xp= np.zeros((X_height+2*padding, X_height+2*padding))
        Xp[padding:(-1-padding+1), padding:(-1-padding+1)]= X
        X= Xp
    S= np.zeros((S_height, S_width), dtype=np.float64)

    for i, y in enumerate(range(0, X_height, stride)):
        for j, x in enumerate(range(0, X_width, stride)):
            S[i, j] = np.sum(X[y:y+Y_height, x:x + Y_width]*Y)
    return S

# Lectura de imagen
im = plt.imread('poligono.png')
im= np.mean(im, axis=2)
hF= np.array([[1, 0, -1],
              [2, 0, -2],
              [1, 0, -1]], dtype=np.float64)
vF= np.array([[1 ,  2,  1],
              [0 ,  0,  0],
              [-1, -2, -1]], dtype=np.float64)

hIm= conv2D(im, hF, padding=1, stride=2)
vIm= conv2D(im, vF, padding=1, stride=2)
sol_Im= np.sqrt(hIm**2 + vIm**2)
print(hIm.shape)

plt.subplot(1, 2, 1)
plt.imshow(im)
plt.subplot(1, 2, 2)
plt.imshow(sol_Im)
