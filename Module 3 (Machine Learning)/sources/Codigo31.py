"""
@author: manupc
Ejemplo de uso de pandas: Visualización de datos con pandas
"""

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
import pandas.plotting as pdplt

# Cargamos los datos
data= pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/geyser.csv")


# Gráfica de líneas
cols= pd.Series(data.waiting)
cols.plot()

#Todas las gráficas (de líneas)
data.plot()

# histograma
data.hist()

# Nubes de puntos
X= pd.Series(data.duration)
Y= pd.Series(data.waiting)
data.plot.scatter(x='duration', y='waiting')

# Cajas y bigotes
data.plot.box()


# Matriz de correlación
pdplt.scatter_matrix(data)