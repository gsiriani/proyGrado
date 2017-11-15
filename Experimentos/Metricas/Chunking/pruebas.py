# -*- coding: utf-8 -*-
path_proyecto = '/home/guille/proyecto/proyGrado'

import sys
sys.path.append(path_proyecto)

import pandas as pd
import numpy as np
from codecs import open, BOM_UTF8
import csv
from keras.preprocessing.sequence import pad_sequences

window_size = 11 # Cantidad de palabras en cada caso de prueba
unidades_ocultas_capa_3 = 33

archivo_embedding = path_proyecto + "/embedding/embedding_total.txt"
archivo_lexicon = path_proyecto + "/embedding/lexicon_total.txt"
archivo_corpus_entrenamiento = path_proyecto + '/corpus/Sentencia_truncada/Entrenamiento/srl_simple_training.csv'
archivo_corpus_pruebas = path_proyecto + '/corpus/Sentencia_truncada/Pruebas/srl_simple_pruebas.csv'

print 'Cargando casos de prueba...' 

# Abro el archivo con casos de prueba
x_train_a = []
x_train_b = []
y_train = []
with open(archivo_corpus_pruebas, 'rb') as archivo_csv:
    lector = csv.reader(archivo_csv, delimiter=',')
    for linea in lector:
        x_train_a.append([int(x) for x in linea[2:-unidades_ocultas_capa_3]])
        y_train.append([int(x) for x in linea[-unidades_ocultas_capa_3:]])


        distancia_palabra = range(-int(linea[0]),len(linea)-(2+unidades_ocultas_capa_3+int(linea[0])))
        distancia_palabra += [np.iinfo('int32').min for _ in range(len(distancia_palabra),50)]
        distancia_verbo = range(-int(linea[1]),len(linea)-(2+unidades_ocultas_capa_3+int(linea[1])))
        distancia_verbo += [np.iinfo('int32').min for _ in range(len(distancia_verbo),50)]
        x_train_b.append(np.array([ np.array([x,y]) for (x,y) in zip(distancia_palabra, distancia_verbo) ]))


#x_train_a = [l[2:] for l in x_train]
#x_train_b = [ [[i-l[0], i-l[1]] for i in range(len(l)-2)] for l in x_train] # Matriz que almacenara distancias a la palabra a analizar
x_train_a = pad_sequences(x_train_a, padding='post', value=0)#CAMBIAR POR INDICE OUT
x_train_b = np.array(x_train_b)
print x_train_b.shape
#y_train = np.array(y_train)




