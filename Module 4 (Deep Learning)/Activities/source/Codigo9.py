#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:07:41 2023

@author: manupc
"""

import numpy as np
import matplotlib.pyplot as plt


# Convolución 1D
X= np.array([1, 1, 2, 2, 3, 3, 2, 2, 1, 1], dtype=np.float64)
Y= np.array([0.5, 1.0, 0.5])

def conv1D(X, Y):
    Conv_len = len(X) - len(Y) + 1

    Y_inv = Y[::-1].copy()
    result = np.zeros(Conv_len, dtype=np.float64)
    for i in range(Conv_len):
        result[i] = np.dot(X[i: i + len(Y)], Y_inv)
    return result


S= conv1D(X, Y)
plt.plot(X)
plt.plot(Y)
plt.plot([None, *S])
plt.legend(['X', 'Y', 'S'])
plt.show()



# Convolución 2D
def conv2D(X, Y):

    Y_height, Y_width = Y.shape
    X_height, X_width = X.shape
    S_height = (X_height - Y_height)+1 #// stride + 1
    S_width = (X_width - Y_width)+1 #// stride + 1

    S= np.zeros((S_height, S_width), dtype=np.float64)

    for y in range(0, S_height):
        for x in range(0, S_width):
            S[y, x] = np.sum(X[y:y+Y_height, x:x + Y_width]*Y)
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

hIm= conv2D(im, hF)
vIm= conv2D(im, vF)
sol_Im= np.sqrt(hIm**2 + vIm**2)
print(hIm.shape)

plt.subplot(1, 2, 1)
plt.imshow(im)
plt.subplot(1, 2, 2)
plt.imshow(sol_Im)
