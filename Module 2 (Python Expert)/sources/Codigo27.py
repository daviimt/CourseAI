"""
@author: manupc
Ejemplo de uso de pandas: indexación y filtrado

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

# Se puede cambiar el índice por defecto a cualquier columna
# Por ejemplo, usaremos la columna kind como índice
data_aux= data.set_index('kind')
print('Cambio de índice a columna kind: ')
print(data_aux)

# Seleccionar las filas con tipo long, sólo la columna duration
tipos= ['long']
columnas= ['duration']
data2= data_aux.loc[tipos, columnas]
print(data2)


# Filtrado con condiciones
# Selección de los datos con duración entre 3 y 3.5
print('Erupciones con duración entre 3 y 3.5: ')
data3= data[(data.duration>=3) & (data.duration<=3.5)]
print(data3)

# También podemos conocer los valores concretos, sin repetir,
# que hay en una columna
print('Valores posibles de la columna kind: ')
print(data.kind.unique())


# Agrupación de datos por kind
data4= data.groupby('kind')[['duration', 'waiting']].mean()
print(data4)