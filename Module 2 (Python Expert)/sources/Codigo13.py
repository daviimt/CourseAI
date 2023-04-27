"""
@author: manupc
Ejemplo de uso de listas
"""

L= [] # Lista vacía

# Comprobación de lista vacía
if L:
    print('Esto nunca se ejecuta')
else:
    print('La lista L está vacía')

# Creación de lista con for e iteración en lista
L2= [x for x in range(5, 10, 2)]
for elemento in L2:
    print(elemento)

# Append/pop
L2.append(27)
print('L2 tras append: ', L2)

L2.pop(0)
print('L2 tras pop(0): ', L2)

print('Ahora la lista tienen ', len(L2), 'elementos')

# Ordenación
L2.append(2)
print('L2 antes de ordenar: ', L2)
L2.sort()
print('L2 tras ordenar', L2)

# Ejemplo de objeto referenciado
L3= L2 # NO SE COPIA. Se referencia a la misma lista
L3.append(88) # Se añade a la misma lista L2, L3
print('L2 tras modificar L3', L2)

# Copia bien hecha
L3= L2.copy() # SE COPIA. Se referencia a listas diferentes
L3.append(101) # Se añade a la misma lista L2, L3
print('L2 tras modificar L3 en copia', L2)
print('L3 tras modificar L3 en copia', L3)


# Listas de listas
L= []
for fila in range(3):
    L.append([])
    for columna in range(3):
        L[fila].append(3*fila+columna)
print('L creada en bucle', L)

# El código anterior equivale a:
L= [[0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
    ]
print('L creada manualmente', L)
