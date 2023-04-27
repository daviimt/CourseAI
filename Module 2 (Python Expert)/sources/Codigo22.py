"""
@author: manupc
Ejemplo de uso de numpy: Operaciones básicas de arrays
"""

import numpy as np


# Suma, resta, multiplicación, división
a= np.array([1, 2, 3])
b= np.array([4, 5, 6])
c= a+b
print('a+b= ', c)
c= a-b
print('a-b= ', c)
c= a*b
print('a*b= ', c)
c= a/b
print('a/b= ', c)
c= a**b
print('a**b= ', c)

# Producto escalar
c= np.dot(a,b)
print('a.dot b: ', c)

# Producto de matrices
a= a.reshape(1, 3)
b= b.reshape(3, 1)
c= a@b
print('a@b: ', c)


# Traspuesta
c= a.T
print('a.t: ', c)

# Inversa
a= np.array([1, 2, 3, 4]).reshape((2, 2))
print('Inv a: ', np.linalg.inv(a))

# Determinante
print('det a', np.linalg.det(a))
