


class Vehiculo:
    def __init__(self,color,nRuedas):
        self.color=color
        self.nRuedas=nRuedas

    def __str__(self):
        return 'vehiculo '+self.color
    
    def mover(self):
        raise Exception('Funcion no implementada')
    
class Coche(Vehiculo):
    def __init__(self,color,nRuedas,cilindrada,nPuertas):
        super().__init__(color,nRuedas)
        self.cilindrada=cilindrada
        self.nPuertas=nPuertas
    
    def __str__(self):
        return 'coche '+self.color
    
    def mover(self):
        print('Coche ',self.color,'se mueve y tiene',self.cilindrada,'cilindradas')
    
    def abrirCcerrar(self):
        print('El coche tiene',self.nPuertas,'puertas',)

class Bicicleta(Vehiculo):
    def __init__(self,color,nRuedas,tipo):
        super().__init__(color,nRuedas)
        self.tipo=tipo

    def __str__(self):
        return 'bici '+self.color
    
    def mover(self):
        print('Bici',self.color,'se mueve y tiene',self.nRuedas,'ruedas')
    


class BicicletaElectrica(Bicicleta):
    def __init__(self,color,nRuedas,tipo,potencia,autonomia):
        super().__init__(color,nRuedas,tipo)
        self.potencia=potencia
        self.autonomia=autonomia
        
    def __str__(self):
        return 'bici electrica '+self.color
    
    def mover(self):
        print('Bici Electrica',self.color,'se mueve y tiene',self.nRuedas,'ruedas')
    
    def bateriaRestante(self):
        print('Bici Electrica',self.color,'se mueve y tiene',self.autonomia,'autonomia')

coche=Coche('Rojo',4,3000,5)
bici=Bicicleta('Rojo',2,'Montana')
biciElec=BicicletaElectrica('Rojo',2,'Montana',400,100)

print(coche)
print(bici)
print(biciElec)

coche.mover()
coche.abrirCcerrar()

bici.mover()

biciElec.mover()
biciElec.bateriaRestante()
    