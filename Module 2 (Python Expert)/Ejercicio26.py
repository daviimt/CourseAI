import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importar datos del archivo CSV
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url)
# Información general del DataFrame
df.info()

# Contar la cantidad de hombres, mujeres y niños
hombres = df[df['sex'] == 'male']['sex'].count()
mujeres = df[df['sex'] == 'female']['sex'].count()
ninos = df[df['age'] < 18]['age'].count()

print("Cantidad de hombres: ", hombres)
print("Cantidad de mujeres: ", mujeres)
print("Cantidad de niños: ", ninos)
