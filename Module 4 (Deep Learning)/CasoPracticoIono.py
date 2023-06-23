import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Cargar el dataset
ionosphere = pd.read_csv('Module 4 (Deep Learning)\ion.csv')

X = ionosphere.drop("Class", axis=1)
y = ionosphere["Class"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, train_size=0.75)

# Definir las columnas numéricas y categóricas
features_num = X_train.select_dtypes(include='number').columns.tolist()
features_cat = X_train.select_dtypes(include='object').columns.tolist()

# Crear transformadores para los datos numéricos y categóricos
transformer_num = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
transformer_cat = make_pipeline(SimpleImputer(strategy="most_frequent"))

# Combinar transformadores en un preprocesador de columnas
preprocessor = make_column_transformer(
    (transformer_num, features_num),
    (transformer_cat, features_cat)
)

# Preprocesar los datos de entrenamiento y prueba
X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)

# Definir el modelo de clasificación (puedes cambiarlo por otro modelo si deseas)
model = LogisticRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Predecir las clases de los datos de prueba
y_pred = model.predict(X_valid)
x_pred = model.predict(X_train)

label_mapping = {'good': 1, 'bad': 0}

y_train_pred = model.predict(X_train)
y_train_pred = [label_mapping[value] for value in y_train_pred]
y_train = [label_mapping[value] for value in y_train]

train_accuracy = roc_auc_score(y_train, y_train_pred)
print("Acurazy train:", train_accuracy)

# Para el conjunto de prueba
y_test_pred = model.predict(X_valid)
y_test_pred = [label_mapping[value] for value in y_test_pred]
y_test = [label_mapping[value] for value in y_valid]

test_accuracy = roc_auc_score(y_test, y_test_pred)
print("Acurazy test:", test_accuracy)
