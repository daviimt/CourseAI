
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import numpy as np
import matplotlib.pyplot as plt

# Cargamos los datos
data= pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/geyser.csv")
print(data.head())

# Preprocesamiento: Cambio de etiquetas a dígitos
data['label']= data.kind.map({'long' : 0, 'short':1})
print(data.head())

# Nos quedamos con los datos que queremos 
data= data[['duration', 'waiting', 'label']].to_numpy()

# Mostramos los datos duration y waiting en colores
class0= data[data[:, 2]==0, :-1]
class1= data[data[:, 2]==1, :-1]

## Límites de la figura
duration_min, duration_max = data[:, 0].min() - 1, data[:, 0].max() + 1
waiting_min, waiting_max = data[:, 1].min() - 1, data[:, 1].max() + 1

plt.scatter(class0[:, 0], class0[:, 1],
            color='red', marker='o', label='long')
plt.scatter(class1[:, 0],class1[:, 1],
            color='blue', marker='x', label='short')

plt.xlim(duration_min, duration_max)
plt.ylim(waiting_min, waiting_max)
plt.show()

# División en train y test
x_train, x_test, y_train, y_test = \
    train_test_split(data[:, :-1], data[:, -1], 
                     test_size=0.25)

# Creamos modelo de regresión logística
model = RandomForestClassifier(n_estimators= 5)

# Lo entrenamos
model.fit(x_train, y_train)

# Comprobamos predicciones en training
train_p= model.predict(x_train)
print('Predicciones correctas en training (predicción, correcto):\n', np.vstack((train_p, y_train)).T)
score = model.score(x_train, y_train)
print('Accuracy en training: ', score)


# Comprobamos predicciones en test
test_p= model.predict(x_test)
print('Predicciones correctas en test (predicción, correcto):\n', np.vstack((test_p, y_test)).T)
score = model.score(x_test, y_test)
print('Accuracy en test: ', score)



# Mostramos parámetros del clasificador
for i in model.estimators_:
    plt.figure(figsize=(15, 15))
    plot_tree(i, class_names=['long', 'short'])


# Mostramos gráficamente con mejor visualización
markers = ('s', 'x', 'o', '^', 'v') # Marcadores a usar
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan') # COlores a usar

# Mapa de colores
plt.figure()
from matplotlib.colors import ListedColormap
cmap = ListedColormap(colors[:2]) # Cogemos mapa para 2 clases

# Superficie de decisión
## Límites de la figura
duration_min, duration_max = data[:, 0].min() - 1, data[:, 0].max() + 1
waiting_min, waiting_max = data[:, 1].min() - 1, data[:, 1].max() + 1

## Creamos grid
resolucion= 0.02
xx1, xx2 = np.meshgrid(np.arange(duration_min, duration_max, resolucion),
                           np.arange(waiting_min, waiting_max, resolucion))

# Predecimos para cada valor del mesh su clase
Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape) # Lo ponemos en forma de mesh

# Mostramos contorno y aplicamos límites
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

    
## Salidas reales
plt.scatter(class0[:, 0], class0[:, 1],
            color='red', marker='o', label='long')
plt.scatter(class1[:, 0],class1[:, 1],
            color='blue', marker='x', label='short')


# Matriz de confusión
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, test_p, labels=model.classes_)
fig = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=model.classes_)
fig.plot()