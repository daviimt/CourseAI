#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 09:21:09 2023

@author: manupc
"""

import tensorflow as tf
import numpy as np

##############################################3
# Creación de tensores

# Creación de tensor constante desde listas Python
a= tf.constant([1, 2, 3], shape=(3,), dtype=tf.float32)
#a[0]= 3 # Esta línea produce un error. a es constante
print('Mi primer tensor constante: ', a)


# Creación de tensor variable desde listas Python
b= tf.Variable([1, 2, 3], shape=(3,), dtype=tf.float32)
#b[0]= 3 # Esto sígue sin poderse hacer
b.assign([3, 2, 1]) # Se asigna el tensor al completo
print('Mi primer tensor variable: ', b)


# Creación de tensor desde NumPy
c_n= np.array([1, 2, 3], dtype=np.float32)
c_t= tf.convert_to_tensor(c_n)
print('Tensor desde Numpy: ', c_t)


# Creación de ndarray desde Tensor
c_n2= c_t.numpy()
c_t= tf.convert_to_tensor(c_n)
print('NDArray desde tensor: ', c_n2)


# Creación de tensor de ceros y unos
z= tf.zeros(shape=(3,2))
o= tf.ones(shape=(2))
e= tf.eye(num_rows=3)
print('Tensor de ceros: ', z)
print('Tensor de unos: ', o)
print('Matriz identidad: ', e)

# Cambio de tipo: casting
e_i= tf.cast(e, tf.int64)
print('Tensor Eye tras casting a entero: ', e_i)

# Cambio de forma de un tensor
e_0= tf.reshape(e, shape=(1, 9))
print('Ejemplo de reshape: ', e_0)

# Eliminación de dimensiones sin usar
e_0_sq= tf.squeeze(e_0)
print('Ejemplo de squeeze: ', e_0_sq)



