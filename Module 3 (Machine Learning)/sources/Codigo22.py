"""

@author: manupc
Dataset: Diabetes

Ejemplo de ajuste automático de hiperparámetros
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


X, y = load_iris(return_X_y=True)

# Vamos a ajustar el mejor modelo de SVC cambiando:
# Tipo de kernel
# Importancia de la regularización(mejora de la generalización del modelo)
# Coeficiente gamma para los kernels rbf, poly, sigmoid

grid = {'kernel': ['linear', 'rbf', 'poly'], 'C':[1, 100, 1000],
        'gamma': [1e-3, 1e-4]}

model = SVC(gamma="scale")
gridSearch = GridSearchCV(model, grid, cv=5) # k-fold Cross-v con K=5
gridSearch.fit(X, y)

# Creaxción de dataframe para comprobar accuracy 
df = pd.concat([pd.DataFrame(gridSearch.cv_results_["params"]),
                pd.DataFrame(gridSearch.cv_results_["mean_test_score"],
                              columns=["Accuracy"])], axis=1)
print('Búsqueda exhaustiva con Grid:')
print(df)



###########################
# Ahora con Random Search
from sklearn.model_selection import RandomizedSearchCV

model = SVC(gamma="scale")

iteraciones = 5 # Cuántas búsquedas queremos hacer
randomSearch = RandomizedSearchCV(model, param_distributions=grid,
                                   n_iter=iteraciones)

randomSearch.fit(X, y)

# Creaxción de dataframe para comprobar accuracy 
df = pd.concat([pd.DataFrame(randomSearch.cv_results_["params"]),
                pd.DataFrame(randomSearch.cv_results_["mean_test_score"],
                              columns=["Accuracy"])], axis=1)
print('Búsqueda aleatoria:')
print(df)
