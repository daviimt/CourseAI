#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:55:45 2023

@author: manupc
"""

import tensorflow as tf
import pandas as pd
import tensorflow.keras.optimizers as optim
import tensorflow.keras.losses as losses

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Cargamos los datos
data= pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")


# Preprocesamiento: Cambio de etiquetas a dígitos y a OneHot
data['label']= data.species.map({'setosa' : 0, 'versicolor':1, 'virginica': 2})
Y= pd.get_dummies( data['label'] ).to_numpy() # Pasamos a One-Hot encoding
X= data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy()

# Creación de modelo con 2 capas ocultas con 10 neuronas cada una, activación relu
# La capa de salida da la probabilidad de cada clase
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])


# Compilación del modelo: Selección de algoritmo de entrenamiento y métrica a usar
model.compile(optimizer= optim.RMSprop( learning_rate= 0.01),
              loss=losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# Entrenamiento:
# Un total de 100 iteraciones con tamaños de batch de 50
Y= tf.convert_to_tensor(Y, dtype=tf.int64)
X= tf.convert_to_tensor(X)

model.fit(X, Y, batch_size=50, epochs=100)

# Obtención de accuracy manualmente
Yp= model.predict(X)
Yp_numeric= tf.argmax(Yp, axis=1)
Y_numeric= tf.argmax(Y, axis=1)
iguales= tf.cast( (Yp_numeric==Y_numeric), tf.float32)
accuracy= tf.reduce_mean(iguales)*100
print('Tasa de aciertos: ', accuracy)

