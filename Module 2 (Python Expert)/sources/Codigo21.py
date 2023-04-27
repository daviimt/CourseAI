"""
@author: manupc
Ejemplo de uso de numpy: Indexación de arrays
"""

import numpy as np


# Indice 1D
a= np.array([1, 2, 3])
print('Array[0]: ', a[0])

# Indice 2D
a= np.array([[1, 2, 3], [4, 5, 6]])
print('Array[1][2]: ', a[1, 2])

# Fila 1 (que no la primera)
b= a[1, :]
print('Fila 1: ', b, 'con shape', b.shape)

# Columna 0 (la primera)
b= a[:, 0]
print('Columna 0: ', b, 'con shape', b.shape)

# Todos los elementos salvo el primero y el último
a= np.arange(10)
print('Todos los elementos salvo el primero y el último: ', a[1:-1])

# Todos los elementos a partir de la posición 2
print('Todos los elementos a partir de la posición 2: ', a[2:])

# Todos los elementos hasta la posición 2
print('Todos los elementos hasta la posición 2: ', a[:2])