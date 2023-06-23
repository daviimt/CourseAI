import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer

hotel_data = pd.read_csv('Module 4 (Deep Learning)\ion.csv')

# Separar características y variable objetivo
X = hotel_data.drop('Class', axis=1)
y = hotel_data['Class']

# Preprocesamiento de características
numeric_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(include='object').columns

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (SimpleImputer(strategy='most_frequent'), categorical_features)
)

X = preprocessor.fit_transform(X)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, train_size=0.75)

# Crear la estructura del modelo utilizando la API funcional de Keras
input_shape = X_train.shape[1]

inputs = tf.keras.Input(shape=(input_shape,))
x = tf.keras.layers.Dense(16, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compilar y entrenar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=200,
    callbacks=[early_stopping],
)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['accuracy', 'val_accuracy']].plot(title="Accuracy")

model.summary()
# Evaluar el modelo en los conjuntos de entrenamiento y prueba
train_accuracy = model.evaluate(X_train, y_train, verbose=0)[1]
test_accuracy = model.evaluate(X_valid, y_valid, verbose=0)[1]

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
