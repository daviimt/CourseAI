"""
@author: manupc
Ejemplo de uso de pandas: creación de data frames
"""

import pandas as pd

## Creación de un data frame por filas
persona = [
    {"nombre": "Perico Palotes", "edad": 27, "pelo": 'castaño'},
    {"nombre": "Mariana Quesada", "edad": 22, "pelo": 'rubio'},
]
persona = pd.DataFrame(persona)
print('Dataframe por filas:\n', persona)


## Creación de un data frame por columnas
persona = {
    "nombre": ["Perico Palotes", "Mariana Quesada"],
    "edad": [27, 22],
    "pelo": ['castaño', 'rubio']
    }
persona = pd.DataFrame(persona)
print('Dataframe por columnas:\n', persona)


## Carga de un dataframe desde fichero CSV
persona= pd.read_csv('DataframePersona.csv')
print('Dataframe desde CSV:\n', persona)