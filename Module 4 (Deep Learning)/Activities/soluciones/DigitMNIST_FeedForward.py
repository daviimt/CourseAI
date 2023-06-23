#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:01:48 2023

Ejercicio: Realizar un clasificador con una red feedforward para
poder distinguir entre 10 tipos de dígitos manuscritos

@author: manupc
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Cargamos el dataset Digits
(xTrain, yTrain), (xTest, yTest)= mnist.load_data()
NumClasses= len(np.unique(yTrain))
print('Tamaño del dataset para entrenamiento: ', len(xTrain))
print('Tamaño del dataset para test: ', len(xTest))
print('Estructura de los datos de entrada (1 imagen): ', xTrain[0].shape)
print('Una imagen de ejemplo:\n', xTrain[0])
print('\n\n')
print('Estructura de los datos de salida: ', yTrain.shape, 'con tipo', type(yTrain), ' y cada componente', yTrain.dtype)
print('Valores posibles de los datos de salida: ', NumClasses)

#######################################
# mostramos algunas imagenes aleatorias
fig, ax = plt.subplots(5, 5)
altoImagen= xTrain[0].shape[0]
anchoImagen= xTrain[0].shape[1]

# Cogemos 25 imágenes al azar del conjunto y las ponemos en una rejilla de 5x5
ImgSeleccionadas= xTrain[ np.random.permutation(len(xTrain))[:25] ].reshape(5, 5, altoImagen, anchoImagen)

# Las imprimimos
for fila in range(5):
    for columna in range(5):
        ax[fila, columna].imshow(ImgSeleccionadas[fila, columna], cmap='gray') # Mostrar imagen en escala de grises
plt.show()



#######################################
### PREPROCESAMIENTO REQUERIDO
# Cada imagen está en una matriz de altoImagen filas por anchoImagen columnas. Las redes feedforward sólo
# pueden tener como entrada patrones en formato array.


# Paso 1: Transformar cada imagen (altoImagen, anchoImagen) 
# a un array de longitud altoImagen*anchoImagen
# PISTA: Usar reshape




# Paso 2: Mirar el rango de los datos (valores mínimo y máximo)
# Transformar los datos para que estén en el rango [0, 1]



# Paso 3: Las salidas son valores categóricos (números enteros). Hay que pasarlos a codificación OneHot
# PISTA: Se puede usar el OneHotEncoder de sklearn, el get_dummies de pandas, o la función:
#         tensorflow.keras.utils.to_categorical. Ejemplo: tf.keras.utils.to_categorical(yTrain, num_classes=10)




#######################################
### Creación del modelo

# Crear un modelo que tenga: 
#   1 capa oculta densa de 512 neuronas con activación relu (tf.keras.layers.Dense)
#   1 capa oculta densa de 128 neuronas con activación relu
#   1 capa de salida densa con tantas neuronas como posibles salidas tenga la red, con actifación softmax



######################################
### Selección de función de coste/error/pérdida
# Es un problema de clasificación multiclase. Una loss function válida sería la entropía cruzada para multiclase
#  o también llamada "Categorical Cross-Entropy". Se encuentra en:
#         tf.keras.losses.CategoricalCrossentropy()



########################################
### Selección de algoritmo de aprendizaje
# Usaremos el Adam por su buen funcionamiento en general. Se encuentra en:
#         tf.keras.optimizers.Adam
# Usar el argumento de entrada "learning_rate" para darle una tasa de aprendizaje de 0.001



######################################### 
# Compilar el modelo con los argumentos optimizer= algoritmo de aprendizaje, y loss=función de pérdida
#  Se debe llevar la cuenta para la métrica 'accuracy' (argumento de entrada metrics)





##########################################
# Entrenamiento

# paso 1: Pasar los datos de entrada y salida a tensores. 
#            Los xTrain y xTest deben ser de tipo float32 o float64
#            Los yTrain e yTest deben ser int8, int16, int32 o int64 porque así lo requiere la función de pérdida



# Paso 2: Fitting. 
#  Aplicar el método fit del modelo para que optimice un total de 20 épocas (argumento de entrada epoch),
#   con una división en batches de tamaño 128 (argumento de entrada batch_size)




################################################
# Validación
# Se puede usar el método .predict para predecir las salidas de las entrada y calcular accuracy a mano
# Se puede también usar el método .evaluate(x,y) para que nos devuelva la métrica dada al compilar el modelo


