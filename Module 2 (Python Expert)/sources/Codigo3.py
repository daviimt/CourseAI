"""
@author: manupc
Ejemplo de sentencia if..else
"""

print('Dime un número entero: ')
a= int( input() )

if a < 0:
    print('El número es negativo')
else: 
    print('El número no es negativo')


# Asignación condicionada
A= 8.5
B= 'Aprobado' if A>=5 else 'Suspenso'
print('El alumno está: ', B)

print('FIN')
