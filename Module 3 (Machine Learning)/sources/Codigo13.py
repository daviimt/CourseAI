#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 09:38:57 2023

@author: manupc
"""

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Carga del conjunto de datos: dígitos
X,y = load_digits(return_X_y=True)
print('Tamaño del dataset: ', X.shape)

# Mostrar algunos
fig, ax = plt.subplots(4, 4)

plt.gray()
for i in range(4):
    for j in range(4):
        
        ax[i, j].imshow(X[i*4+j].reshape(8,8))

plt.show()


# División en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.25)


# Creación de modelo
model = LogisticRegression(fit_intercept=True,
                        multi_class='auto',
                        penalty='l2', #ridge regression
                        solver='saga',
                        max_iter=10000)

model.fit(X_train, y_train)



test_p= model.predict(X_test)
cm = confusion_matrix(y_test, test_p, labels=model.classes_)
fig = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=model.classes_)
fig.plot()

# Métricas del clasificador
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc_test= accuracy_score(y_test, test_p)
precision_test= precision_score(y_test, test_p, average='micro')
recall_test= recall_score(y_test, test_p, average='micro')
f1_test= f1_score(y_test, test_p, average='micro')

print('Los resultados EN TEST son:')
print('Accuracy: ', acc_test)
print('Precision: ', precision_test)
print('Recall: ', recall_test)
print('F1-Score: ', f1_test)