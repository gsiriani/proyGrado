#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import uniform
import pandas as pd
import numpy as np

window_size = 11 # Cantidad de palabras en cada caso de prueba
unidades_ocultas_capa_2 = 300
unidades_ocultas_capa_3 = 16 # SE MODIFICA PARA CADA PROBLEMA A RESOLVER

archivo_corpus_entrenamiento = "./csv_prueba.csv"



# Entreno
print 'Cargando casos de entrenamiento...'

# Abro el archivo con casos de entrenamiento
df = pd.read_csv(archivo_corpus_entrenamiento, delim_whitespace=True, skipinitialspace=True, header=None, quoting=3)

# a=df.at[0,0]
# print a
# b=eval(a)
# print b
# for x in b:
#     print x[0]


# Separo features de resultados esperados
largo = len(df)
x_train = np.array(df.iloc[:largo,:1])
y_train = np.array(df.iloc[:largo,1:])

print len(x_train)
x_train_a=[]
x_train_b=[]
for i in range(len(x_train)):
	print x_train[i,0]
	print len(x_train[i,0])
	x_train_a.append([palabra for (palabra,distancia) in eval(x_train[i,0])])
	print x_train_a	
	x_train_b.append([distancia for (palabra,distancia) in eval(x_train[i,0])])
	print x_train_b
