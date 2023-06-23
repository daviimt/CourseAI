#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:13:28 2023

@author: manupc
"""

import tensorflow as tf


##############################################3
# Operaciones con tensores
a= tf.random.uniform(shape=(2, 3))
b= tf.random.uniform(shape=(2, 3))

c= tf.add(a, b)
d= a-b
e= tf.multiply(a, b)
f= tf.reduce_mean(a)
g= tf.reduce_mean(a, axis=1)
h= tf.transpose(b)
i= tf.linalg.matmul(a, h)
print('Suma de tensores', c)
print('Resta de tensores', d)
print('Multiplicación de tensores', e)
print('Media de tensor', f)
print('Media de tensor en eje 1', g)
print('Traspuesta de tensor', h)
print('Multiplicación matricial', i)