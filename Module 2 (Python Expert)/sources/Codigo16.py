"""
@author: manupc
Ejemplo de uso de diccionarios
"""

Dvacio= {}
if Dvacio:
    print('Esto nunca se muestra')
else:
    print('El diccionario está vacío')
    
# Definición de diccionario explícitamente
D= {'11223344A' : ('Perico', 8.5),
    '22334455B' : ('María', 5.1)}

# Mostrar claves mediante iteración y mediante keys
print('Iteración sobre las claves del diccionario')
for clave in D:
    print(clave)
print('Acceso a claves mediante keys(): ', D.keys())

# Insercción de nuevo ítem en el diccionario
D['33445566C']= ('Juan', 8.9)

# Iteración sobre diccionario:
for clave in D:
    elemento= D[clave]
    print(elemento[0], 'tiene DNI', clave, 
          'y calificación', elemento[1])