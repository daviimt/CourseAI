"""
@author: manupc
Ejemplo de sentencia for
"""

coleccion= [2, 3.7, 'algo']

for dato in coleccion:
    print('En la colección está el dato', dato)

for i, dato in enumerate(coleccion):
    print('Posición ', i+1)
    print('En la colección está el dato', dato)



print('FIN')
