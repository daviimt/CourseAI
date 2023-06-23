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
X= tf.Variable(3.0)

with tf.GradientTape() as tape:
    Y= X**2

# Cálculo de la derivada en X=3
derivada= tape.gradient(Y, X)
print('La derivada en x=', X.numpy(), 'es', derivada.numpy())


############################################
# Ejemplo: cálculo del mínimo de una función
# (Usando descenso del gradiente)
Num_itera= 100 # Número de iteraciones
lmbda= 0.1 # Tasa de aprendizaje
X0= tf.squeeze( tf.random.uniform(shape=(1,)) )*10
X= tf.Variable(X0)
print('Punto inicial X0=', X0.numpy())

# Función a optimizar
def FuncionAOptimizar(X):
    return (X-1)**2 + 2

for i in range(Num_itera):
    
    # Forward pass
    with tf.GradientTape() as tape:
        Y= FuncionAOptimizar(X)
    DX= tape.gradient(Y, X) # Coger gradientes
    print('En iteración', i+1, 'obtengo gradiente', DX.numpy())
    X= tf.Variable(X-lmbda*DX) # Regla del descenso del gradiente
    print('y actualizo el punto a X=', X.numpy())

print('El mínimo de la función está en aproximadamente X=', X.numpy())