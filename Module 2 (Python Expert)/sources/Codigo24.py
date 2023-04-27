"""
@author: manupc
Ejemplo de uso de numpy: Números aleatorios
"""

import numpy as np

# Fijamos semilla aleatoria para reproducibilidad
np.random.seed(12345)


# U(0,1)
a= np.random.rand(2, 3) # Matriz de 2,3
print('U(0,1): ', a)

# N(0,1)
a= np.random.randn(3, 2) # Matriz de 3,2
print('N(0,1): ', a)


# Enteros aleatorios
a= np.random.randint(0, 11, 3) # 3 enteros aleatorios de 0 a 10
print('enteros de 0 a 10: ', a)


# Permutación aleatoria
a= np.random.permutation(5) # Permutación de 5 elementos
print('permutación: ', a)
