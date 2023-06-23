#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:01:48 2023

Ejercicio: Realizar un clasificador con una red CNN para
poder distinguir entre 10 tipos de imágenes. Estos son:
    aviones, automóviles, pájaros, gatos, renos, perros, raanas, caballos, barcos, camiones
    
OJO: Este dataset es EN COLOR. 

@author: manupc
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Cargamos el dataset Digits
(xTrain, yTrain), (xTest, yTest)= cifar10.load_data()
NumClasses= len(np.unique(yTrain))
print('Tamaño del dataset para entrenamiento: ', len(xTrain))
print('Tamaño del dataset para test: ', len(xTest))
print('Estructura de los datos de entrada (1 imagen): ', xTrain[0].shape)
print('Una imagen de ejemplo:\n', xTrain[0])
print('\n\n')
print('Estructura de los datos de salida: ', yTrain.shape, 'con tipo', type(yTrain), ' y cada componente', yTrain.dtype)
print('Valores posibles de los datos de salida: ', NumClasses)

CLASES= ['Avion', 'automóvil', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']

#######################################
# mostramos algunas imagenes aleatorias
fig, ax = plt.subplots(5, 5)
altoImagen= xTrain[0].shape[0]
anchoImagen= xTrain[0].shape[1]
canalesColor= xTrain[0].shape[2]

# Cogemos 25 imágenes al azar del conjunto y las ponemos en una rejilla de 5x5
ImgSeleccionadas= xTrain[ np.random.permutation(len(xTrain))[:25] ].reshape(5, 5, altoImagen, anchoImagen, canalesColor)

# Las imprimimos
for fila in range(5):
    for columna in range(5):
        ax[fila, columna].imshow(ImgSeleccionadas[fila, columna])
plt.show()



#######################################
### PREPROCESAMIENTO REQUERIDO
# Cada imagen está en una matriz de altoImagen filas por anchoImagen columnas y 3 canales.



# Paso 1: Mirar el rango de los datos (valores mínimo y máximo)
# Transformar los datos para que estén en el rango [0, 1]
xTrain= xTrain/255.0
xTest= xTest/255.0



# Paso 3: Las salidas son valores categóricos (números enteros). Hay que pasarlos a codificación OneHot
# PISTA: Se puede usar el OneHotEncoder de sklearn, el get_dummies de pandas, o la función:
#         tensorflow.keras.utils.to_tategorical. Ejemplo: tf.keras.utils.to_categorical(yTrain, num_classes=10)
yTrain= tf.keras.utils.to_categorical(yTrain.squeeze(), num_classes=NumClasses)
yTest= tf.keras.utils.to_categorical(yTest.squeeze(), num_classes=NumClasses)




#######################################
### Creación del modelo

# Crear un modelo que tenga: 
#  - 1 capa convolucional 2D que tenga como entrada (input_shape) una imagen de (altoImage, anchoImagen, canalesColor),
#        que tenga 32 filtros de tamaño (3,3) (argumento kernel_size), con padding same y strides (2,2)
#  - 1 Capa de MaxPooling 2D con padding same, strides de (1,1) y pool_size de (3,3)
#  - 1 Capa convolucional de 64 filtros de tamaño (3,3), con padding same y stride (2,2)
#  - 1 Capa de MaxPooling 2D con padding same, strides de (1,1) y pool_size de (3,3)
#  - 1 capa densa de 256 neuronas con activación relu (¡OJO: No olvidar adaptar la salida de la convolucional antes!)
#  - 1 capa densa de 128 neuronas con activación relu
#  - 1 capa de salida densa con tantas neuronas como clases en el problema y activación softmax
modelo = tf.keras.Sequential([
            tf.keras.layers.Conv2D(input_shape=(32, 32, 3), kernel_size=(3, 3), padding='same', strides=(2, 2), filters=32),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), padding='same', strides=(2, 2), filters=64),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
            ])



######################################
### Selección de función de coste/error/pérdida
# Es un problema de clasificación multiclase. Una loss function válida sería la entropía cruzada para multiclase
#  o también llamada "Categorical Cross-Entropy". Se encuentra en:
#         tf.keras.losses.CategoricalCrossentropy()
lossF= tf.keras.losses.CategoricalCrossentropy()


########################################
### Selección de algoritmo de aprendizaje
# Usaremos el Adam por su buen funcionamiento en general. Se encuentra en:
#         tf.keras.optimizers.Adam
# Usar el argumento de entrada "learning_rate" para darle una tasa de aprendizaje de 0.001
opt= tf.keras.optimizers.Adam(learning_rate=0.001)


######################################### 
# Compilar el modelo con los argumentos optimizer= algoritmo de aprendizaje, y loss=función de pérdida
#  Se debe llevar la cuenta para la métrica 'accuracy' (argumento de entrada metrics)
modelo.compile(optimizer= opt, loss=lossF, metrics=['accuracy'])




##########################################
# Entrenamiento

# paso 1: Pasar los datos de entrada y salida a tensores. 
#            Los xTrain y xTest deben ser de tipo float32 o float64
#            Los yTrain e yTest deben ser int8, int16, int32 o int64 porque así lo requiere la función de pérdida
txTrain= tf.convert_to_tensor(xTrain)
tyTrain= tf.convert_to_tensor(yTrain, dtype=tf.int8)
txTest= tf.convert_to_tensor(xTest)
tyTest= tf.convert_to_tensor(yTest, dtype=tf.int8)



# Paso 2: Fitting. 
#  Aplicar el método fit del modelo para que optimice un total de 20 épocas (argumento de entrada epoch),
#   con una división en batches de tamaño 128 (argumento de entrada batch_size)
modelo.fit(txTrain, tyTrain, batch_size=128, epochs=20)



################################################
# Validación
# Se puede usar el método .predict para predecir las salidas de las entrada y calcular accuracy a mano
# Se puede también usar el método .evaluate(x,y) para que nos devuelva la métrica dada al compilar el modelo
_, accTr= modelo.evaluate(txTrain, tyTrain)
print('La métrica de accuracy en train es: ', accTr)

_, accTs= modelo.evaluate(txTest, tyTest)
print('La métrica de accuracy en test es: ', accTs)

