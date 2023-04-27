"""
@author: manupc
Ejemplo de uso de pandas: creación de data frames
"""

import pandas as pd

"""
 Conjunto de datos a cargar: Old Faithful geyser dataset.
 Contiene datos de erupciones del geiser Old Faithful. 
 Sus atributos son:
     
     duración: Tiempo de duración de la erupción
     Tiempo de espera: Tiempo entre una erupción y la siguiente
     Tipo: Erupción corta o larga
     El dataset se encuentra en: 
         https://github.com/mwaskom/seaborn-data/blob/master/geyser.csv
     Su acceso a través de Internet se puede hacer mediante:
         https://raw.githubusercontent.com/mwaskom/seaborn-data/master/geyser.csv
"""

import pandas as pd


# Cargamos los datos
data= pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/geyser.csv")

print('Las 5 primeras filas: ')
print( data.head(5) ) # Mostramos las 5 primeras filas de la tabla

print('Las 5 últimas filas: ')
print( data.tail(5) ) # Mostramos las 5 últimas filas de la tabla

print('10 filas al azar: ')
print( data.sample(10) ) # Mostramos 10 filas al azar de la tabla


# Tamaño del dataset
print('El dataset tiene ', data.shape[0], 'filas y', data.shape[1], 'columnas')


# Información de tipos de valores y valores nulos/perdidos
print('Información de tipos de valores y valores nulos/perdidos: ')
print(data.info())

# Información de estadística descriptiva genérica
print('Estadística descriptiva básica: ')
print(data.describe())


# Ordenación por Tipo y duración
data= data.sort_values(['kind', 'duration'])
print('Datos ordenados por tipo y duración: ')
print(data)