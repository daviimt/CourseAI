"""
@author: manupc
Ejemplo de uso de pandas: Operaciones básicas con columnas


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


# Suma y promedio de los valores de duration
print('Suma de duration: ', data.duration.sum() )
print('Promedio de duration: ', data.duration.mean() )

# Creación de columna con la cumsum de waiting
cumsum_waiting= data.waiting.cumsum()
print('Suma acumulada de waiting: ', cumsum_waiting)

# Inserción de nueva columna en el dataframe
data['CumSum_Waiting']= cumsum_waiting
print('Tabla tras añadir nueva columna')
print(data)