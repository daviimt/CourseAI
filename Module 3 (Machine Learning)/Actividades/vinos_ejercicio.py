#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 17:31:46 2023

Se trata de un conjunto de vinos italianos de la misma región, de 3 productores diferentes.
Se realizó un análisis químico de las muestras, y el objetivo consiste en saber si
es posible predecir cuál es el productor en base a los análisis químicos (alcohol, ácido málico,
Ceniza, alcalinidad de la ceniza, magnesio, etc.)

@author: manupc
"""
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np

# Cargamos el dataset de vinos y vemos sus atributos
dataset= load_wine()
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

# Número de clases:
print('Número de clases a predecir: ', len(df.target.unique()), 'que son:', df.target.unique())



# ¿Es necesario quitar valores perdidos? Si es el caso, hacerlo aquí:

    
# ¿Se debería hacer alguna transformación de los datos? Si es es caso, hacerlo aquí:


# División ALEATORIA en training (50%) y test (50%) . Ejemplo:


    
# ¿Qué modelo se va a usar para aprender? Instanciarlo aquí y entrenarlo en TRAIN


# Comprobación del Score en training y en test




# ¿Y si aprendemos con otro modelo? Haz la prueba e instancialo aquí y entrenarlo en TRAIN


# Comprobación del Score en training y en test
