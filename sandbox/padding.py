path_proyecto = '/home/guille/proyGrado'

import sys
sys.path.append(path_proyecto)
import pandas as pd
from script_auxiliares import print_progress
import numpy as np


archivo_corpus_entrenamiento = path_proyecto + '/corpus/Oracion/Entrenamiento/ner_training.csv'

# Abro el archivo con casos de entrenamiento
df = pd.read_csv(archivo_corpus_entrenamiento, delim_whitespace=True, skipinitialspace=True, header=None, quoting=3)
largo = len(df)
x_test = np.array(df.iloc[:largo,:1])
maximo=0

for f in range(largo):		    
    print_progress(f, largo, prefix = 'Progreso:', suffix = 'Completado', bar_length = 50)
    oracion = eval(x_test[f,0])
    local=len([palabra for (palabra,distancia) in oracion])
    maximo=max(maximo,local)

print_progress(largo, largo, prefix = 'Progreso:', suffix = 'Completado', bar_length = 50)
print maximo
