#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:13:28 2023

@author: manupc
"""

import tensorflow as tf
import numpy as np


##############################################3
# Indexación con tensores
a= tf.random.normal(shape=(1000,))


# Indexación booleana
b= (a>=0) # Creación de array bool
print('Elementos no negativos: ', tf.reduce_sum(tf.cast(b, dtype=tf.int64)))
c= a[b] # Indexación booleana
print('Elementos no negativos con indexación: ', c.shape)

# Indexación con :
d= a[1:10]
print('Los elementos del segundo al décimo: ', d)


# En tensores multidimensionales:
a= tf.random.normal(shape=(1000,2))


# Indexación booleana
b= (a>=0) # Creación de array bool
print('Elementos no negativos: ', tf.reduce_sum(tf.cast(b, dtype=tf.int64)))
c= a[b] # Indexación booleana
print('Elementos no negativos con indexación: ', c.shape)

# Indexación con :
d= a[1:10]
print('Los elementos del segundo al décimo: ', d)


##############################################3
# División con split
a= tf.squeeze( tf.random.uniform(shape=(999, 1)) )
splits= tf.split(a, num_or_size_splits=3)
print('Splits:', len(splits))
print('Segundo split: ', splits[1])


##############################################3
# Concatenación con concat
a= tf.random.uniform(shape=(2,)) 
b= tf.random.uniform(shape=(3,)) 
c= tf.concat([a,b], axis=0)
print('concat(a,b):', c)


##############################################3
# Apilación con stack
a= tf.random.uniform(shape=(3,)) 
b= tf.random.uniform(shape=(3,)) 
c= tf.stack([a,b], axis=0)
d= tf.stack([a,b], axis=1)
print('stack(a,b) en axis=0:', c)
print('stack(a,b) en axis=1:', d)


##############################################3
# Selección con gather
a= tf.convert_to_tensor([[0.1, 0.2],
                         [0.3, -0.4],
                         [0.5, -0.6],
                         [0.7, 0.8]])

b= tf.gather_nd(a, indices = [2, 1])
print('gather (elemento)', b)
c= tf.gather_nd(a, indices = [[2], [1]])
print('gather (tensor)', c)