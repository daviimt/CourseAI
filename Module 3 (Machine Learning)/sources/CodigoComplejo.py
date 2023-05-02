"""
@author: manupc
Ejemplo de creación de clases: Clase complejo
"""

from math import sqrt

class Complejo:
    def __init__(self, real= 0, imag= 0):
        self.Real= real
        self.Imag= imag
        
    def Conjugado(self):
        return Complejo(self.Real, -self.Imag)
    
    def Abs(self):
        return sqrt(self.Real**2 + self.Imag**2)
    
    def Suma(self, otro):
        return Complejo(self.Real + otro.Real, self.Imag + otro.Imag)
    
    def Resta(self, otro):
        return Complejo(self.Real - otro.Real, self.Imag - otro.Imag)
    

    def __str__(self): # Función automática para transformar a cadena
        return str(self.Real) + '+' + str(self.Imag)+"i"

