# -*- coding: utf-8 -*-
path_proyecto = '/home/guille/proyGrado'

import sys
sys.path.append(path_proyecto)

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten
from keras.initializers import TruncatedNormal, Constant, RandomUniform
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from vector_palabras import palabras_comunes
from random import uniform
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from script_auxiliares import print_progress
import time
from codecs import open, BOM_UTF8
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score


def main(tarea, precalculado = False):

	archivo_embedding = path_proyecto + "/embedding/embedding_ordenado.txt"
	archivo_lexicon = path_proyecto + "/embedding/lexicon_total.txt"
	archivo_corpus_pruebas = path_proyecto + '/corpus/Ventana/Testing/' + tarea + '_testing.csv'


	def cargarCasos(archivo):

		# Abro el archivo
		df = pd.read_csv(archivo, sep=',', skipinitialspace=True, header=None, quoting=3)
		largo = len(df)

		# Separo features de resultados esperados
		x = np.array(df.iloc[:largo,:11])
		y = np.array(df.iloc[:largo,11:])

		return x, y


	# Cargo casos
	inicio_carga_casos = time.time()

	print 'Cargando casos de prueba...' 
	x_test, y_test = cargarCasos(archivo_corpus_pruebas)

	duracion_carga_casos = time.time() - inicio_carga_casos

	window_size = 11 # Cantidad de palabras en cada caso de prueba
	vector_size = 150 if precalculado else 50 # Cantidad de features a considerar por palabra
	unidades_ocultas_capa_2 = 300
	unidades_ocultas_capa_3 = len(y_test[0])

	
	archivo_best = 'mejores_pesos.hdf5'

	log = 'Log de ejecucion:\n-----------------\n'
	log += '\nTESTING'
	log += '\nTarea: ' + tarea
	log += '\nModelo de red: Ventana'
	log += '\nEmbedding inicial: '
	if precalculado:
		log += 'Precalculado'
	else:
		log += 'Aleatorio'
	log += '\nOptimizer: adam'
	log += '\nActivacion: relu'

	print 'Compilando red...'

	# Defino las capas de la red

	# Cargo embedding inicial

	if precalculado:
		embedding_inicial = []
		for l in open(archivo_embedding):
		    embedding_inicial.append(list([float(x) for x in l.split()])) 

		embedding_inicial = np.array(embedding_inicial)

		cant_palabras = len(embedding_inicial)

		embedding_layer = Embedding(input_dim=cant_palabras, output_dim=vector_size, weights=[embedding_inicial],
		                            input_length=window_size, trainable=True)

	else:
		palabras = palabras_comunes(archivo_lexicon) # Indice de cada palabra en el diccionario

		cant_palabras = len(palabras)  # Cantidad de palabras consideradas en el diccionario

		embedding_layer = Embedding(input_dim=cant_palabras, output_dim=vector_size,
		                            embeddings_initializer=RandomUniform(minval=-0.05, maxval=0.05, seed=1),
		                            input_length=window_size, trainable=True)


	second_layer = Dense(units=unidades_ocultas_capa_2,
	                     use_bias=True,
	                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=2),
	                     bias_initializer=Constant(value=0.1))

	third_layer = Dense(units=unidades_ocultas_capa_3,
	                    use_bias=True,
	                    kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=3),
	                    bias_initializer=Constant(value=0.1))


	# Agrego las capas al modelo
	model = Sequential()
	model.add(embedding_layer)
	model.add(Flatten())
	model.add(second_layer)
	model.add(Activation("relu"))
	model.add(third_layer)
	model.add(Activation("softmax"))

	# Compilo la red
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()


	# Obtengo metricas
	print 'Obteniendo metricas...'
	inicio_metricas = time.time()

	model.load_weights(archivo_best)

	etiquetas = range(unidades_ocultas_capa_3)
	predictions = model.predict(x_test, batch_size=200, verbose=0)
	y_pred = []
	for p in predictions:
	    p = p.tolist()
	    ind_max = p.index(max(p))
	    etiqueta = etiquetas[ind_max]
	    y_pred.append(etiqueta)
	y_true = []
	for p in y_test:
	    p = p.tolist()
	    ind_max = p.index(max(p))
	    etiqueta = etiquetas[ind_max]
	    y_true.append(etiqueta)
	conf_mat = confusion_matrix(y_true, y_pred, labels=etiquetas)
	accuracy = accuracy_score(y_true, y_pred)
	(precision, recall, fscore, _) = precision_recall_fscore_support(y_true, y_pred)

	duracion_metricas = time.time() - inicio_metricas


	# list all data in history
	log += '\n\nTiempo de carga de casos de Prueba: {0} hs, {1} min, {2} s'.format(int(duracion_carga_casos/3600),int((duracion_carga_casos % 3600)/60),int((duracion_carga_casos % 3600) % 60))
	log += '\nDuracion del calculo de metricas: {0} hs, {1} min, {2} s'.format(int(duracion_metricas/3600),int((duracion_metricas % 3600)/60),int((duracion_metricas % 3600) % 60))

	np.set_printoptions(threshold=10000, edgeitems=1000, linewidth=100000)

	log += '\n\nAccuracy: ' + str(accuracy)
	log += '\nPrecision: ' + str(precision)
	log += '\nRecall: ' + str(recall)
	log += '\nMedida-F: ' + str(fscore)

	log += '\n\nMatriz de confusion:\n' + str(conf_mat)

	#print log
	open("log_testing.txt", "w").write(BOM_UTF8 + log)


if __name__ == '__main__':
	precalculado= False if (sys.argv[2] is None or sys.argv[2] == 'a') else True
	main(str(sys.argv[1]), precalculado)