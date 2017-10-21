# -*- coding: utf-8 -*-
path_proyecto = '/home/guille/proyecto/proyGrado'

import sys
sys.path.append(path_proyecto)

from vector_palabras import palabras_comunes
import pandas as pd
from script_auxiliares import print_progress
import time
from codecs import open, BOM_UTF8

unidades_ocultas_capa_3 = 17

archivo_embedding = path_proyecto + "/embedding/embedding_ordenado.txt"
archivo_lexicon = path_proyecto + "/embedding/lexicon_total.txt"
archivo_corpus_entrenamiento = path_proyecto + '/corpus/Ventana_indizada/Entrenamiento/ner_training.csv'
archivo_corpus_pruebas = path_proyecto + '/corpus/Ventana_indizada/Pruebas/ner_pruebas.csv'

# Cargo embedding inicial
palabras = palabras_comunes(archivo_lexicon) # Indice de cada palabra en el diccionario

indice_OUT = palabras.obtener_indice("OUT")
print 'El indice de OUT es: ' + str(indice_OUT)

inicio_carga_casos = time.time()
print 'Cargando casos de prueba...' 

# Abro el archivo con casos de prueba
df = pd.read_csv(archivo_corpus_pruebas, sep=',', skipinitialspace=True, header=None, quoting=3)
largo = len(df)


print 'Separo casos por oraciones'
oraciones = []
oracion_actual = {'x':[], 'y':[], 'largo':0}
for caso in df.values:
	print caso[6]
	oracion_actual['x'].append(caso[:11])
	oracion_actual['y'].append(caso[11:])
	oracion_actual['largo']+=1
	if (caso[6] == indice_OUT):
		print 'OUT detectado'
		oracion_actual['x']=np.array(oracion_actual['x'])
		oracion_actual['y']=np.array(oracion_actual['y'])
		oraciones.append(oracion_actual)
		oracion_actual = {'x':[], 'y':[], 'largo':0}

duracion_carga_casos = time.time() - inicio_carga_casos

print 'Cantidad de casos calculados: ' + str(sum([o['largo'] for o in oraciones]))
print 'Cantidad de casos esperados:  ' + str(largo)