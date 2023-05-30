import pandas as pd

tabla={
    "Caso":[1,2,3,4,5,6,7,8,9],
    "Tasacion":["+","-","+","-","-","+","+","+","+"],
    "Valoracion":["-","-","+","+","+","+","+","-","+"],
    "Entradas":["No","Si","No","Si","Si","Si","Si","Si","No"],
    "Balance":["+","-","+","+","-","-","+","-","-"],
    "Aprobar":["No","No","Si","Si","No","Si","Si","No","No"]
}

data=pd.DataFrame(tabla)

print(data.head)
print(data.tail)
print(data.sample)
print(data.shape[0])
print(data.info)