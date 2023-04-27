"""
@author: manupc
Ejemplo de uso de numpy: Creación de arrays
"""

import numpy as np


# Array desde listas
a= np.array([[1, 2, 3], 
             [4, 5, 6]])
print('Array: ', a)
print('Dimensiones: ', a.ndim)
print('Número de elementos: ', a.size)
print('Shape: ', a.shape)

# Array de 0'sy 1's
ceros= np.zeros(3)
print('Array de ceros: ', ceros)
unos= np.ones(3)
print('Array de unos: ', unos)

# Array vacío (basura)
basura= np.empty((2, 2))
print('Array vacío: ', basura)

# Array con arange y reshape
arange= np.arange(6).reshape(2, 3)
print('Arange con reshape: ', arange)

# Array con linspace
linsp= np.linspace(0, 1, 5)
print('Array con linspace: ', linsp)