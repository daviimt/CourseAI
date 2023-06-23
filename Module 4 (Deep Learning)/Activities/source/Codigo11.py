#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:07:41 2023

@author: manupc
"""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Carga del dataset Fashion MNIST
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('El dataset contiene ', len(train_images), 'imágenes de entrenamiento')
print('El tamaño de cada imagen es de ', train_images.shape[1], 'por', train_images.shape[1])


# El dataset no se proporciona de forma correcta. Hay que darle 1 dimensión de channel
train_images = train_images.reshape((train_images.shape[0], 
                                     train_images.shape[1], 
                                     train_images.shape[2], 1))
test_images = test_images.reshape((test_images.shape[0], 
                                     test_images.shape[1], 
                                     test_images.shape[2], 1))

# Además cada pixel es un valor de 0..255. Lo transformamos a [0,1]
train_images= train_images.astype('float32')/255.0
test_images= train_images.astype('float32')/255.0


# Mostramos algunas imágenes
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# Creación de capa convolucional 2D con:
#  10 filtros
#  Cada filtro de tamaño 3x3
convL= keras.layers.Conv2D(filters= 10, kernel_size=(3,3))

model= keras.Sequential([convL])

Yp= model.predict(train_images)
print('Tamaño de la salida de la capa: ', Yp.shape)


# Vemos el número de parámetros de entrenamiento:
from keras.utils.layer_utils import count_params  
num_param= count_params(model.trainable_weights)
print('La red tiene parámetros a entrenar: ', num_param)