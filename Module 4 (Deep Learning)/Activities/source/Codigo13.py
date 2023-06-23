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
test_images= test_images.astype('float32')/255.0




# Creación de red neuronal convolucional de ejemplo
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', 
                 input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)) )
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

print('Resumen del modelo')
print(model.summary())


opt = keras.optimizers.SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, 
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])



model.fit(train_images, train_labels, epochs=10, batch_size=32,
          validation_data=(test_images, test_labels))

Yp= model.predict(train_images)
Yp_label= tf.argmax(Yp, axis=1)
Y_label= train_labels
aciertos= (Y_label==Yp_label)
aciertos= tf.cast(aciertos, 'float32')
print('Accuracy manual en training: ', tf.reduce_mean(aciertos))

Yp= model.predict(test_images)
Yp_label= tf.argmax(Yp, axis=1)
Y_label= test_labels
aciertos= (Y_label==Yp_label)
aciertos= tf.cast(aciertos, 'float32')
print('Accuracy manual en test: ', tf.reduce_mean(aciertos))