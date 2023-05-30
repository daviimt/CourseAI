import numpy as np

# Generar 10000 pares de números aleatorios
points= np.random.rand( 10000, 2 )*2 - 1 #U(-1, 1)
# Calcular la norma de cada par de números
norms = np.linalg.norm(points, axis=1)

# Filtrar los puntos cuya norma es mayor que 1
mask = norms <= 1
filtered_points = np.sum(mask)

# Aproximar PI utilizando los puntos restantes
pi_approx = 4 * filtered_points / 10000

print(f"PI se aproxima a: {pi_approx}")
