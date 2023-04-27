"""
@author: manupc
Ejemplo de definición de funciones con return y yield
"""
def MaximoMinimo(dato1, dato2):
    
    if dato1 > dato2:
        Maximo= dato1
        Minimo= dato2
    else:
        Maximo= dato2
        Minimo= dato1
    return Minimo, Maximo


def Contador(ValorFin):
    contador= 0
    while contador < ValorFin:
        yield contador
        contador= contador+1


# Comienza el programa

## Ejemplo de MaximoMinimo
A= 32
B= 27
ValorMinimo, ValorMaximo= MaximoMinimo(A, B)
print('Salida de MaximoMinimo: ', ValorMinimo, ValorMaximo)

## Ejemplo de Contador
for dato in Contador(7):
    print(dato)

