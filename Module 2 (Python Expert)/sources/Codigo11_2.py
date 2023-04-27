"""
@author: manupc
Ejemplo de uso de pass y raise
"""

def UnaFuncion(parametros=None):
    pass

UnaFuncion() # No da error


def OtraFuncion(parametros=None):
    raise Exception('Error: Función aún no implementada')
    
OtraFuncion() # Ejecutar esta línea da error