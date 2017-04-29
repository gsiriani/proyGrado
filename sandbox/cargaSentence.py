path_proyecto = '/home/guille/proyGrado'

import sys
sys.path.append(path_proyecto)

from random import uniform
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from script_auxiliares import print_progress

window_size = 11 # Cantidad de palabras en cada caso de prueba
unidades_ocultas_capa_2 = 300
unidades_ocultas_capa_3 = 16 # SE MODIFICA PARA CADA PROBLEMA A RESOLVER

archivo_corpus_entrenamiento = path_proyecto + "/sandbox/csv_prueba.csv"



# Entreno
print 'Cargando casos de entrenamiento...'

# Abro el archivo con casos de entrenamiento
df = pd.read_csv(archivo_corpus_entrenamiento, delim_whitespace=True, skipinitialspace=True, header=None, quoting=3)

a=df.at[0,0]
b=eval(a)
print b




