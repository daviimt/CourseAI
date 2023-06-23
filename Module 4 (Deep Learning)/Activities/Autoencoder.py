#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:04:45 2023

@author: manupc
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential


# Carga del dataset
(x_train, _), (x_test, _) = fashion_mnist.load_data()


# Transformación de rango de datos a [0,1]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print ('Tamaño del conjunto de train: ', x_train.shape)
print ('Tamaño del conjunto de test; ', x_test.shape)

##########################################
# Creación del modelo autoencoder

# 1. Creación del modelo como secuencia encoder-decoder:
#   1.1. Encoder: Una única capa densa con 64 neuronas y activación ReLU
#        OJO!!!: Tenemos imágenes de entrada, hay que hacer un flatten() primero!
#   1.2. Decoder: Una única cada con 28*28=784 neuronas con activación sigmoid
#        OJO!!: La salida tiene que ser una imagen, hay que aplicar una capa reshape a tamaño 28x28


##########################################
# Entrenamiento

# 1. Función de pérdida: Error cuadrático medio (MSE)

# 2. Optimizador: Adam, con learning_rate=0.001

# 3. Compilación del modelo


# 4. Fit del modelo

############################################3
# Test

# 1. Aplicar el modelo al conjunto de test (método .predict)


# 2. Hacer un subplots de 2 filas y 5 columnas con matplotlib. 
# 3. Visualizar las 5 primeras imágenes del test en una fila (las entradas)
# 4. Visualizar la reconstrucción de las 5 primeras imágenes del test en otra fila (las salidas)




