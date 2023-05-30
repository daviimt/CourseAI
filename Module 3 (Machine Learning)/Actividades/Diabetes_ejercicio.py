#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:14:50 2023

Se trata de un conjunto de pacientes de diabetes, a los que se les hace un seguimiento anual.
Se desea conocer la progresión de la condición (valor numérico a estimar) en base 
a características como edad en años, sexo, índice de masa corporal, etc.
El problema consiste en estimar el valor de la progresión (problema de regresión)



@author: manupc
"""

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np

# Cargamos el dataset de cáncer de mama y vemos sus atributos
dataset= load_diabetes()
for key in dataset:
    print(key)

# Mostramos descripción (Descomentar para ver)
#print(dataset['DESCR'])

# Convertimos a pandas juntando entradas (data) y salidas (target)
df= pd.DataFrame(data= dataset['data'], columns= dataset['feature_names'])
df['target']= pd.Series(dataset['target'])

# Muestra de los datos
print('Muestra de los datos:')
print(df.sample(10))

# Descripción de los datos
print('Descripción de los datos:')
print(df.describe())

# Información de los datos
print('Información de los datos:')
print(df.info())




# ¿Es necesario quitar valores perdidos? Si es el caso, hacerlo aquí:

    
# ¿Se debería hacer alguna transformación de los datos? Si es es caso, hacerlo aquí:


# División ALEATORIA en training (50%) y test (50%) . Ejemplo:
datos= df.to_numpy()

X= datos[:, :-1] # Entradas: Todas las columnas salvo la última
Y= datos[:, -1].astype(int) # Salidas: La última columna

# División en train y test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, shuffle=True)


    
# ¿Qué modelo se va a usar para aprender? Instanciarlo aquí y entrenarlo en TRAIN
# Ejemplo: Probar una red neuronal MLPRegressor con learning_rate_init=0.1

# Comprobación del MSE (Error Cuadrático medio) en training y en test
# Se calcula como np.mean( (salida_modelo_con_predict - salida_real)**2 )



# Otro ejemplo: 
# Probar ahora con un DecisionTreeRegressor

# Comprobación del MSE (Error Cuadrático medio) en training y en test
# Se calcula como np.mean( (salida_modelo_con_predict - salida_real)**2 )


# ¿Qué diferencia se observa en la comparación de los errores de train en ambos
# modelos? ¿Y en la de test?