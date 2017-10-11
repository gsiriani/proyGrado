# -*- coding: utf-8 -*-
path_proyecto = '/home/guille/proyecto/proyGrado'

import sys
sys.path.append(path_proyecto)

import time
import numpy as np
import csv
from codecs import open, BOM_UTF8

unidades_ocultas_capa_3 = 16

archivo_corpus_entrenamiento = path_proyecto + '/corpus/Oracion/Entrenamiento/ner_training.csv'
archivo_corpus_pruebas = path_proyecto + '/corpus/Oracion/Pruebas/ner_pruebas.csv'

inicio_carga_casos = time.time()
print 'Cargando casos de prueba...' 

# Abro el archivo con casos de prueba
x_test = []
y_test = []
casos_test = []
with open(archivo_corpus_pruebas, 'rb') as archivo_csv:
    lector = csv.reader(archivo_csv, delimiter=',')
    for linea in lector:
    	casos_test.append([int(x) for x in linea])
        x_test.append([int(x) for x in linea[:-unidades_ocultas_capa_3]])
        y_test.append([int(x) for x in linea[-unidades_ocultas_capa_3:]])
        if len(linea)==142+unidades_ocultas_capa_3:
        	print linea



print 'Separo casos por oraciones'
oraciones = []
oracion_actual = {'x_a':[], 'x_b':[], 'y':[], 'largo':0}
primera = True

for caso in casos_test:	
	oracion_actual['x_a'].append(caso[1:-unidades_ocultas_capa_3])	
	oracion_actual['x_b'].append([[i-caso[0]] for i in range(len(caso[1:-unidades_ocultas_capa_3])-1)])
	oracion_actual['y'].append(caso[-unidades_ocultas_capa_3:])
	oracion_actual['largo']+=1
	if (caso[0]+1 == len(caso[1:-unidades_ocultas_capa_3])): # Guardo oracion anterior			
		oracion_actual['x_a']=np.array(oracion_actual['x_a']) # En teoria NO requiere PADDING porque todos los casos miden lo mismo		
		oracion_actual['x_b']=np.array(oracion_actual['x_b']) 
		oracion_actual['y']=np.array(oracion_actual['y'])
		oraciones.append(oracion_actual)
		oracion_actual = {'x_a':[], 'x_b':[], 'y':[], 'largo':0}

duracion_carga_casos = time.time() - inicio_carga_casos

print 'Cantidad de casos calculados: ' + str(sum([o['largo'] for o in oraciones]))
print 'Cantidad de casos esperados:  ' + str(len(casos_test))
print 'Cantidad de oraciones: ' + str(len(oraciones))
print 'Largo promedio de oraciones: ' + str(np.mean([o['largo'] for o in oraciones]))
print 'Oracion mas larga: ' + str(np.max([o['largo'] for o in oraciones]))
print 'Oracion mas corta: ' + str(np.min([o['largo'] for o in oraciones]))


