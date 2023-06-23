#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:59:23 2023

@author: manupc
"""


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Cargamos la imagen
img= image.load_img('gato.jpg')
img = image.img_to_array(img)


# Mostramos algunos
plt.figure()
plt.imshow(img/255.0)
plt.show()

# La preprocesamos para que tenga el tamaño de entrada a la red
img= image.smart_resize(img, size=(224, 224))[np.newaxis,]
print('Tamaño de los datos de entrada: ', img.shape)



# Creación del modelo
model = ResNet50(weights='imagenet')

# Preprocesamiento de entrada
img = preprocess_input(img)
plt.figure()
plt.imshow(img[0])

# Predicciones
preds = model.predict(img)
print('Los 3 más probablos son:', decode_predictions(preds, top=3)[0])

