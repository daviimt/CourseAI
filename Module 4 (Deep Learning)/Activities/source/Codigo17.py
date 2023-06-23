#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:22:16 2023

@author: manupc
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Cargamos datos (no nos interesal los labels, es aprendizaje "no supervisado")
(x_train, _), (x_test, _) = mnist.load_data()

# Preprocesamiento: Escala a [0,1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Añadir nueva dimensión para el canal
x_train=  np.expand_dims(x_train, axis=-1)
x_test=  np.expand_dims(x_test, axis=-1)


num_pixels= np.prod(x_train.shape[1:])
print('Cada imagen tiene ', num_pixels, 'pixeles')

# Encoder de la red
encoder= keras.Sequential()
encoder.add(layers.Flatten())
encoder.add(layers.Dense(16, activation='relu')) # Resultado: Latent Space

# Decoder de la red
decoder= keras.Sequential()
decoder.add(layers.Dense(num_pixels, activation='relu'))
decoder.add(layers.Reshape(x_train.shape[1:]))



# Modelo completo: Encoder/Decoder
model= keras.Sequential()
model.add(encoder)
model.add(decoder)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss=keras.losses.MeanSquaredError())

model.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True)

# Probamos predicción
y_test= model.predict(x_test)

plt.figure(figsize=(15, 5))
for i in range(5):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[i])
    plt.subplot(2, 5, 5+i+1)
    plt.imshow(y_test[i])