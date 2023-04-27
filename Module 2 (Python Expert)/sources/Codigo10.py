"""
@author: manupc
Ejemplo de definición de funciones
"""
def Saludo(nombre= None):
    
    if nombre is not None:
        print("Hola, amigo ", nombre)
    else:
        print("No me has dicho tu nombre")


# Comienza el programa
print('Primera llamada: ')
Saludo()
print('Segunda llamada: ')
Saludo("Manuel")
print('Tercera llamada: ')
Saludo(nombre= "Lolo")

print('Otra llamada más: ')
Saludo(nombre= None)