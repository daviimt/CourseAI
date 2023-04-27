"""
@author: manupc
Ejemplo de uso de clases: Clase complejo
"""

# Definición de clase padre
class Animal:
    def __init__(self, nombre):
        self.nombre= nombre
    
    def hablar(self):
        raise Exception('Operación hablar no implementada')
    
    def moverse(self):
        raise Exception('Operación moverse no implementada')


# Clase Perro, heredada de Animal
class Perro( Animal ):
    def __init__(self, nombre, color):
        super().__init__(nombre)
        self.color= color
    
    def __str__(self):
        return 'Perro '+ self.nombre+ ' de color '+ self.color
    
    def hablar(self):
        print('Guau! guau!')
    
    def moverse(self):
        print(self.nombre, 'se mueve a 4 patas')
        

# Clase Vaca, heredada de Animal
class Vaca( Animal ):
    def __init__(self, nombre):
        super().__init__(nombre)
    
    def __str__(self):
        return 'Vaca '+ self.nombre
    
    def hablar(self):
        print('Muuuuuu!')
    
    def moverse(self):
        print(self.nombre, 'se mueve a 4 patas')
        

# Clase Abeja, heredada de Animal
class Abeja( Animal ):
    def __init__(self, nombre):
        super().__init__(nombre)
    
    def __str__(self):
        return 'Abeja '+ self.nombre
    
    def hablar(self):
        print('Bzzzzzzz!')
    
    def moverse(self):
        print(self.nombre, 'vuela')
        

    def picar(self):
        print('Auch!')

