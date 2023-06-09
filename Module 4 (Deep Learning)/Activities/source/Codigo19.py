#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 09:26:39 2023

@author: manupc
"""

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Reducimos el dataset para que no tarde tanto
x_train= x_train[:20000]

# Inserción de canal en la imagen
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')

# Normalizar a [-1,1]
x_train = (x_train - 127.5) / 127.5

BUFFER_SIZE = len(x_train)
BATCH_SIZE = 32

# Generamos dataset con datos dispuestos aleatoriamente y batch de tam. 256
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)



# Modelo generador
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model



# Modelo discriminador
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# Función de pérdida
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)



# Función de pérdida del discriminador
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# Función de pérdida del generador
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Algoritmos de aprendizaje
generator= make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# Parámetros de aprendizaje
EPOCHS = 100 # iteraciones a realizar
noise_dim = 100 # Tamaño de la entrada del ruido 
num_examples_to_generate = 16 # Número de muestras "fake" a generar


# Paso de entrenamiento:
@tf.function
def train_step(images):
    # Generación de ruido de entrada al generador
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generamos imágenes
        generated_images = generator(noise, training=True)
        
        # Evaluamos reales y fakes con el discriminador
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        # Funciones de pérdida del generador y del discriminador
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # Actualización de pesos
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))



# Función de entrenamiento general
def train(dataset, epochs):
    for epoch in range(epochs):
        print('Epoca {} de {}'.format(epoch+1, epochs))
        for image_batch in dataset:
            train_step(image_batch)
    
    
train(train_dataset, EPOCHS)


# Generamos imagen de ejemplo
plt.figure(figsize=(15,5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    noise = tf.random.normal([1, noise_dim])
    img = generator(noise)
    img= tf.squeeze(img)
    plt.imshow(img)