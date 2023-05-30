import numpy as np

# Generar un array de números del 0 al 8
array = np.arange(9)
# Reorganizar el array en una matriz de 3x3
matriz = np.reshape(array, (3, 3))
print(matriz)