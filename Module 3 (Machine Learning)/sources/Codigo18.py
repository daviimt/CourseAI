
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
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


# Creamos modelo de regresión logística
logisticRegr = LogisticRegression()


# Definición de las métricas a usar para evaluación
Metricas= ['accuracy', 'precision', 'recall', 'f1']

# K-fold cv con K=10
K= 10
results = cross_validate(estimator=logisticRegr,
                               X=data[:, :-1],
                               y=data[:, -1],
                               cv= K,
                               scoring= Metricas,
                               return_train_score=True)

# Resultado del aprendizaje con CV:
print('Accuracy en train: ', results['train_accuracy'].mean())
print('Precision en train: ', results['train_precision'].mean())
print('Recall en train: ', results['train_recall'].mean())
print('F1-Score en train: ', results['train_f1'].mean())

print('Accuracy en test: ', results['test_accuracy'].mean())
print('Precision en test: ', results['test_precision'].mean())
print('Recall en test: ', results['test_recall'].mean())
print('F1-Score en test: ', results['test_f1'].mean())
