"""
@author: manupc
Ejemplo de uso de numpy: Operaciones básicas de arrays
"""

import numpy as np


a= np.array([[1, 2, 3], [4, 5, 6]])

# Media, sd, varianza, suma, producto, mínimo, máximo
print('mean a', np.mean(a))
print('s.d a', np.std(a))
print('var a', np.var(a))
print('min/max a', np.min(a), np.max(a))
print('argmin/argmax a', np.argmin(a), np.argmax(a))

print('sum a', np.sum(a))
print('prod a', np.prod(a))
print('cumsum a', np.cumsum(a))