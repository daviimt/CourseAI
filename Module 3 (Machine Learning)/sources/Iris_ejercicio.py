import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Cargamos los datos
col_names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Class']
iris =  pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
                    names = col_names)


# ¿Hay valores perdidos? Tratarlos aquí.



# ¿Es necesario hacer preprocesamiento? Hacerlo aquí


# División en train y test


# Selección del algoritmo de aprendizaje


# Cálculo de métricas de evaluación en TRAIN (Homogeneidad y completitud en nuestro caso)

# Cálculo de métricas de evaluación en TEST