"""
@author: manupc
Ejemplo de uso de pandas: Visualización de datos con matplotlib
"""

import matplotlib.pyplot as plt
import numpy as np


###################################
# Gráfico de líneas básico
x= [1, 2, 3, 4]
y1= np.random.rand(4)
y2= np.random.rand(4)

plt.figure() # Nueva figura
plt.plot(x, y1) # Dato 1
plt.plot(x, y2, color='red') # Dato 2

plt.legend(['y_1', 'y_2']) # leyenda de la figura

# Etiqueta de los ejes
plt.xlabel('Valor de X')
plt.ylabel('Valor de Y1/Y2')

# Rejilla (opcional)
plt.grid(visible=True, which='major', color='#888888', linestyle='-')
plt.savefig('figura1.png') # Guardar figura a fichero


###################################
# Gráfico de líneas múltiple
f, ax= plt.subplots(1, 2, figsize=(15, 4)) # 1 fila, 2 columnas
f.suptitle('Figura con subplots') # Título de la figura entera
ax[0].plot(x, y1)
ax[1].plot(x, y2)
ax[0].set_xlabel('X de subfigura 1')
ax[0].set_ylabel('Valor Y1')
ax[1].set_xlabel('X de subfigura 2')
ax[1].set_ylabel('Valor Y2')
plt.ylabel('Valor de Y1/Y2')
plt.savefig('figura2.png') # Guardar figura a fichero



###################################
# Gráfico de dispersión (puntos)
x1= np.random.randn(100)
x2= np.random.randn(100)+1
y1= np.random.randn(100)
y2= np.random.randn(100)+1
plt.figure() # Nueva figura
plt.scatter(x1, y1) # Dato 1
plt.scatter(x2, y2, color='red') # Dato 2
plt.savefig('figura3.png') # Guardar figura a fichero


###################################
# Histogramas
y= np.random.randn(1000)
plt.figure() # Nueva figura
plt.hist(y)
plt.savefig('figura4.png') # Guardar figura a fichero


###################################
# Gráficos de barras vertical
x= ['Uno', 'Dos', 'Tres', 'Cuatro']
y= np.random.randint(10, 101, size=len(x))
plt.figure() # Nueva figura
plt.bar(x, y)
plt.savefig('figura5.png') # Guardar figura a fichero


###################################
# Gráficos de barras horizontal
x= ['Uno', 'Dos', 'Tres', 'Cuatro']
y= np.random.randint(10, 101, len(x))
plt.figure() # Nueva figura
plt.barh(x, y)
plt.savefig('figura6.png') # Guardar figura a fichero


###################################
# Gráficos de sectores
x= ['Uno', 'Dos', 'Tres', 'Cuatro']
y= np.random.randint(0, 101, len(x))
plt.figure() # Nueva figura
plt.pie(y, labels= x)
plt.savefig('figura7.png') # Guardar figura a fichero


###################################
# Gráficos de cajas y bigotes
y= np.random.randn(1000, 2)
plt.figure() # Nueva figura
plt.boxplot(y, labels= ['Uno', 'Dos'])
plt.savefig('figura8.png') # Guardar figura a fichero



###################################
# Gráficos de superficies y contornos
# Generación de datos de ejes de coordenadas
x,y = np.meshgrid(np.linspace(-1,1,15),np.linspace(-1,1,15))

# Generación de datos de eje Z
z= np.cos(x*np.pi)*np.cos(y*np.pi)

f= plt.figure(figsize= (10, 4)) # Nueva figura
ax1 = f.add_subplot(121, projection='3d')
ax1.plot_surface(x,y,z,rstride=1,cstride=1,cmap='viridis')
ax2 = f.add_subplot(122)
cf = ax2.contourf(x,y,z,51,vmin=-1,vmax=1,cmap='viridis')
f.show()
f.savefig('figura9.png') # Guardar figura a fichero
