
from random import uniform
import pandas as pd

window_size = 11 # Cantidad de palabras en cada caso de prueba
unidades_ocultas_capa_2 = 300
unidades_ocultas_capa_3 = 16 # SE MODIFICA PARA CADA PROBLEMA A RESOLVER

archivo_corpus_entrenamiento = "./csv_prueba.csv"



# Entreno
print 'Cargando casos de entrenamiento...'

# Abro el archivo con casos de entrenamiento
df = pd.read_csv(archivo_corpus_entrenamiento, delim_whitespace=True, skipinitialspace=True, header=None, quoting=3)

a=df.at[0,0]
print a
b=eval(a)
print b
for x in b:
    print x[0]




