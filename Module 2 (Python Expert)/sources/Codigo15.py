"""
@author: manupc
Ejemplo de uso de sets
"""

C= { x for x in 'Mañana' }
print(C)

# Añadir nuevo elemento y eliminar elemento
C.add('e')
print('Tras añadir e: ', C)

C.pop()
print('Tras añadir e: ', C)

# Unión, intersección y diferencia
C2= {'a', 'e',  'i', 'o', 'u'}
C3= C.union(C2)
print('Unión de ', C, ' y ', C2, ': ', C3)

C3= C.intersection(C2)
print('Intersección de ', C, ' y ', C2, ': ', C3)

C3= C.difference(C2)
print('Diferencia entre ', C, ' \ ', C2, ': ', C3)

# Relaciones entre conjuntos
print('¿', C3, ' subconjunto de ', C, '?:  ', C3.issubset(C))

# Comprueba si 'a' está en el conjunto C3
a= 'a'
print('¿"a" está en ', C3, '?: ', a in C3) 
