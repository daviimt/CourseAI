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
modelo= Sequential([tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(784, activation='sigmoid'),
                    tf.keras.layers.Reshape((28,28))
                    ])


##########################################
# Entrenamiento

# 1. Función de pérdida: Error cuadrático medio (MSE)
lossF= tf.keras.losses.MSE

# 2. Optimizador: Adam, con learning_rate=0.001
opt= tf.keras.optimizers.Adam(learning_rate=0.001)

# 3. Compilación del modelo
modelo.compile(optimizer= opt, loss= lossF, metrics=['MSE'])


# 4. Fit del modelo
modelo.fit(x_train, x_train, batch_size=128, epochs=10)

############################################3
# Test

# 1. Aplicar el modelo al conjunto de test (método .predict)
salidas= modelo.predict(x_test)


# 2. Hacer un subplots de 2 filas y 5 columnas con matplotlib. 
# 3. Visualizar las 5 primeras imágenes del test en una fila (las entradas)
# 4. Visualizar la reconstrucción de las 5 primeras imágenes del test en otra fila (las salidas)
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(salidas[i])
  plt.title("reconstruida")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()



