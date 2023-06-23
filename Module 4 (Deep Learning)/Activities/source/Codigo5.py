#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:22:15 2023

@author: manupc
"""

import pandas as pd
import tensorflow as tf

# Cargamos los datos
data= pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/geyser.csv")


# Preprocesamiento: Cambio de etiquetas a dígitos
data['label']= data.kind.map({'long' : 0, 'short':1})


X= data[['duration', 'waiting']].to_numpy()
Y= data[['label']].to_numpy()


# Convertimos datos a tensor Dataset
X= tf.data.Dataset.from_tensor_slices(tf.constant(X))
Y= tf.data.Dataset.from_tensor_slices(tf.constant(Y))

# Los unimos en un único dataset
dataset= tf.data.Dataset.zip( (X, Y) )

# mostramos los datos
for patron in dataset:
    print('Entrada: ', patron[0].numpy(), 'salida: ', patron[1].numpy())
    
# Aleatorización de patrones
print('Tras shuffle')
dataset= dataset.shuffle(buffer_size=len(X))
for patron in dataset:
    print('Entrada: ', patron[0].numpy(), 'salida: ', patron[1].numpy())



# Creación de batches de tamaño 5
batch_dataset= dataset.batch(batch_size=10)
num_batches= 0
for batch in batch_dataset:
    num_batches+= 1
    print('Batch', batch)
print('Han salido un total de ', num_batches, 'batches de tamaño 10 para ', len(dataset), 'datos')




#############################################################
# Importación de datasets de Tensorflow
import tensorflow_datasets as tfds
print('Datasets en Tensorflow: ')
print(tfds.list_builders()) #Mostrar los datasets


# Ejemplo: Dataset de MNIST
data_bldr= tfds.builder('mnist')
print(data_bldr.info)

# Descarga del dataset y preparación
data_bldr.download_and_prepare()

# Convertir a dataset (train/test)
dataset= data_bldr.as_dataset()
print('Sub-datasets de MNIST: ', dataset.keys())
trainingData= dataset['train']
testData= dataset['test']

trainingData= trainingData.shuffle(buffer_size= len(trainingData))

# Mostrar algunos elementos del dataset
for i, patron in enumerate(trainingData):
    print('Tam. de imagen de entrada: ', patron['image'].shape, 'y salida asociada:', patron['label'])
    if i>=10:
        break
    
import matplotlib.pyplot as plt

fig= plt.figure(figsize=(15, 4))
for i, patron in enumerate(trainingData):
    if i>=10:
        break
    ax= fig.add_subplot(2, 5, i+1)
    ax.imshow(patron['image'])
plt.show()