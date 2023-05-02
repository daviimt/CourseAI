#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:37:01 2023

@author: manupc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

# Cargamos los datos
data= pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/geyser.csv")
print(data.head())
data= data.to_numpy()

#####################################################
# Codificación de etiquetas (clases)
le= LabelEncoder()

# Ajuste del encoder
le.fit(data[:, -1])

# Clases modeladas
print('Clases modeladas: ', le.classes_)


# Transformación de datos
transformado= le.transform(data[:, -1])

# Mostramos datos transformados
print('Datos transformados: \n', transformado)




#####################################################
# Codificación ordinal
oe= OrdinalEncoder()



# Ajuste del encoder
oe.fit(np.unique(data[:, -1]).reshape(-1, 1))

# Clases modeladas
print('Clases modeladas: ', oe.categories_)


# Transformación de datos
transformado= oe.transform(data[:, -1].reshape(-1, 1))

# Mostramos datos transformados
print('Datos transformados: \n', transformado.reshape(-1))



#####################################################
# Codificación One Hot
ohe= OneHotEncoder()

data= pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
print(data.head())
data= data.to_numpy()

# Ajuste del encoder
ohe.fit(np.unique(data[:, -1]).reshape(-1, 1))


# Clases modeladas
print('Clases modeladas: ', ohe.categories_)


# Transformación de datos
transformado= ohe.transform(data[:, -1].reshape(-1, 1)).toarray()


# Mostramos datos transformados
print('Datos transformados: \n', transformado)

