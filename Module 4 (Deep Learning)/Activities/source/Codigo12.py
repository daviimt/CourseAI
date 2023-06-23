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
num_classes= len(class_names)

print('El dataset contiene ', len(train_images), 'imágenes de entrenamiento')
print('El tamaño de cada imagen es de ', train_images.shape[1], 'por', train_images.shape[1])


# El dataset no se proporciona de forma correcta. Hay que darle 1 dimensión de channel
train_images = train_images.reshape((train_images.shape[0], 
                                     train_images.shape[1], 
                                     train_images.shape[2], 1))
test_images = test_images.reshape((test_images.shape[0], 
                                     test_images.shape[1], 
                                     test_images.shape[2], 1))

# Transformación de las etiquetas a valores categóricos 
print(train_labels[:10])

#train_labels= tf.keras.utils.to_categorical(train_labels)
#test_labels= tf.keras.utils.to_categorical(test_labels)



# Además cada pixel es un valor de 0..255. Lo transformamos a [0,1]
train_images= train_images.astype('float32')/255.0
test_images= train_images.astype('float32')/255.0




# Creación de capa convolucional 2D con:
#  10 filtros
#  Cada filtro de tamaño 3x3
convL= keras.layers.Conv2D(filters= 10, kernel_size=(3,3), 
                           input_shape=(train_images.shape[1], train_images.shape[2], 1))

model= keras.Sequential([convL,
                         keras.layers.Flatten(),
                         keras.layers.Dense(units=100, activation='relu'),
                         keras.layers.Dense(units=num_classes, activation='softmax')])


print('Resumen del modelo')
print(model.summary())

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss= keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']
              )


# Vemos el número de parámetros de entrenamiento:
from keras.utils.layer_utils import count_params  
num_param= count_params(model.trainable_weights)
print('La red tiene parámetros a entrenar: ', num_param)


model.fit(train_images, train_labels, epochs=10, batch_size=32)

Yp= model.predict(train_images)
Yp_label= tf.argmax(Yp, axis=1)
Y_label= train_labels
aciertos= (Y_label==Yp_label)
aciertos= tf.cast(aciertos, 'float32')
print('Accuracy manual: ', tf.reduce_mean(aciertos))