"""
@author: manupc
Ejemplo de uso de pandas: Tratamiento de valores perdidos


 Conjunto de datos a cargar: Food consumption
 Contiene datos de consumo relativo de ciertos alimentos
 en Europa y algunos países escandinavos. Cada valor representa
 el porcentaje de la población que consume dicho alimento.
     El dataset se encuentra en: 
         https://openmv.net/info/food-consumption
     Su acceso a través de Internet se puede hacer mediante:
         https://openmv.net/file/food-consumption.csv
"""

import pandas as pd

# Cargamos los datos
data= pd.read_csv("https://openmv.net/file/food-consumption.csv")
print(data.info()) # Vista general de los datos
print('Tamaño de tabla: ', data.shape)

# Tratamiento de valores perdidos eliminando filas y columnas
print('Tratamiento de valores perdidos eliminando filas:')
data2= data.dropna()
print(data2.info())
print('Tamaño de tabla: ', data2.shape)

print('Tratamiento de valores perdidos eliminando columnas:')
data2= data.dropna(axis=1)
print(data2.info())
print('Tamaño de tabla: ', data2.shape)

print('Tratamiento de valores perdidos rellenando con valor:')
data2= data.fillna(0)
print(data2.info())
print('Tamaño de tabla: ', data2.shape)

print('Tratamiento de valores perdidos rellenando con último valor:')
data2= data.fillna(method='pad')
print(data2.info())
print('Tamaño de tabla: ', data2.shape)

print('Tratamiento de valores perdidos rellenando con siguiente valor:')
data2= data.fillna(method='ffill')
print(data2.info())
print('Tamaño de tabla: ', data2.shape)
