#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 10:56:19 2023

@author: manupc
"""


# Pedimos los datos:
print("Introduzca un número: ")
dato1= float( input() ) # Cogemos el dato como número real
print("Introduzca otro número (distinto de 0): ")
dato2= float( input() )
print('Has introducido ', dato1, ' y ', dato2)
# Definimos resultados
texto= """
=============================
Voy a hacer algunos cálculos
=============================
"""

suma= dato1+dato2
division= dato1/dato2
divisionEntera= dato1/dato2
exponenciacion= dato1**dato2

# Salida
print('La suma es: ', suma)
print('La división es: ', division)
print('La división entera es: ', divisionEntera)
print('La exponenciación es: ', exponenciacion)
