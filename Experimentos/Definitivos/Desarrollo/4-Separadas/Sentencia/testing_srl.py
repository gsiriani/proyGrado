# -*- coding: utf-8 -*-
path_proyecto = '/home/guille/proyGrado'

import sys
sys.path.append(path_proyecto)

from keras.models import Model
from keras.layers import Dense, Activation, Embedding, Flatten, Conv1D, Input, Concatenate
from keras.layers.pooling import GlobalMaxPooling1D
from keras.initializers import TruncatedNormal, Constant, RandomUniform
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from vector_palabras import palabras_comunes
from random import uniform
import csv
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
	archivo_corpus_pruebas = path_proyecto + '/corpus/Sentencia/Testing/' + tarea + '_testing.csv'
	vector_size_distancia = 5 # Cantidad de features para representar la distancia a la palabra a etiquetar
	largo_sentencias = 50


	def cargarCasos(archivo):

		# Abro el archivo

		x = []
		y = []
		with open(archivo, 'rb') as archivo_csv:
		    lector = csv.reader(archivo_csv, delimiter=',')
		    for linea in lector:
		    	largo_x = largo_sentencias + 2
		        x.append([int(t) for t in linea[:largo_x]])
		        y.append([int(t) for t in linea[largo_x:]])

		x_a = [l[2:] for l in x]
		x_b = [ [largo_sentencias+i-l[0] for i in range(largo_sentencias)] for l in x] # Matriz que almacenara distancias a la palabra a analizar
		x_a = np.array(x_a)
		x_b = np.array(x_b)
		x_c = [ [largo_sentencias+i-l[1] for i in range(largo_sentencias)] for l in x] # Matriz que almacenara distancias a la palabra a analizar
		x_c = np.array(x_c)
		y = np.array(y)
		print y[0]

		return x_a, x_b, x_c, y


	# Cargo casos
	inicio_carga_casos = time.time()

	print 'Cargando casos de prueba...' 
	x_test_a, x_test_b, x_test_c, y_test = cargarCasos(archivo_corpus_pruebas)

	duracion_carga_casos = time.time() - inicio_carga_casos

	vector_size = 150 if precalculado else 50 # Cantidad de features a considerar por palabra
	unidades_ocultas_capa_2 = 300
	unidades_ocultas_capa_2_2 = 500
	unidades_ocultas_capa_3 = len(y_test[0])

	
	archivo_best = 'mejores_pesos.hdf5'

	log = 'Log de ejecucion:\n-----------------\n'
	log += '\nTESTING'
	log += '\nTarea: ' + tarea
	log += '\nModelo de red: Convolutiva'
	log += '\nEmbedding inicial: '
	if precalculado:
		log += 'Precalculado'
	else:
		log += 'Aleatorio'
	log += '\nOptimizer: adam'
	log += '\nActivacion: relu'

	print 'Compilando red...'


	# Defino las capas de la red

	main_input = Input(shape=(largo_sentencias,), name='main_input')

	aux_input_layer = Input(shape=(largo_sentencias,), name='aux_input')

	distance_embedding_layer = Embedding(input_dim=largo_sentencias*2, output_dim=vector_size_distancia,
	                            embeddings_initializer=RandomUniform(minval=-0.05, maxval=0.05, seed=4),
	                            trainable=True)(aux_input_layer)

	aux_input_layer2 = Input(shape=(largo_sentencias,), name='aux_input2')

	distance_embedding_layer2 = Embedding(input_dim=largo_sentencias*2, output_dim=vector_size_distancia,
	                            embeddings_initializer=RandomUniform(minval=-0.05, maxval=0.05, seed=5),
	                            trainable=True)(aux_input_layer2)    

	concat_layer_aux = Concatenate()([distance_embedding_layer, distance_embedding_layer2])   

	# Cargo embedding inicial

	if precalculado:
		embedding_inicial = []
		for l in open(archivo_embedding):
		    embedding_inicial.append(list([float(x) for x in l.split()])) 

		embedding_inicial = np.array(embedding_inicial)

		cant_palabras = len(embedding_inicial)

		embedding_layer = Embedding(input_dim=cant_palabras, output_dim=vector_size, weights=[embedding_inicial],
		                            trainable=True)(main_input)

	else:
		palabras = palabras_comunes(archivo_lexicon) # Indice de cada palabra en el diccionario

		cant_palabras = len(palabras)  # Cantidad de palabras consideradas en el diccionario

		embedding_layer = Embedding(input_dim=cant_palabras, output_dim=vector_size,
		                            embeddings_initializer=RandomUniform(minval=-0.05, maxval=0.05, seed=1),
		                            trainable=True)(main_input)                    

	concat_layer = Concatenate()([embedding_layer, concat_layer_aux])

	convolutive_layer = Conv1D(filters=unidades_ocultas_capa_2, kernel_size=5)(concat_layer)

	x_layer = GlobalMaxPooling1D()(convolutive_layer)

	second_layer = Dense(units=unidades_ocultas_capa_2,
	                     use_bias=True,
	                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=2),
	                     bias_initializer=Constant(value=0.1))(x_layer)

	y_layer = Activation("tanh")(second_layer)

	second_layer_2 = Dense(units=unidades_ocultas_capa_2_2,
	                     use_bias=True,
	                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=6),
	                     bias_initializer=Constant(value=0.1))(y_layer)

	y_layer_2 = Activation("tanh")(second_layer_2)

	third_layer = Dense(units=unidades_ocultas_capa_3,
	                    use_bias=True,
	                    kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=3),
	                    bias_initializer=Constant(value=0.1))(y_layer_2)


	softmax_layer = Activation("softmax", name='softmax_layer')(third_layer)


	# Agrego las capas al modelo

	model = Model(inputs=[main_input, aux_input_layer, aux_input_layer2], outputs=[softmax_layer])

	# Compilo la red

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()


	# Obtengo metricas
	print 'Obteniendo metricas...'
	inicio_metricas = time.time()

	model.load_weights(archivo_best)

	etiquetas = range(unidades_ocultas_capa_3)
	predictions = model.predict({'main_input': x_test_a, 'aux_input': x_test_b, 'aux_input2': x_test_c}, batch_size=200, verbose=0)
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

	log += '\n\nPrecision: ' + str(precision)
	log += '\nRecall: ' + str(recall)
	log += '\nMedida-F: ' + str(fscore)

	log += '\n\nMatriz de confusion:\n' + str(conf_mat)

	#print log
	open("log_testing.txt", "w").write(BOM_UTF8 + log)


if __name__ == '__main__':
	precalculado= False if (sys.argv[2] is None or sys.argv[2] == 'a') else True
	main(str(sys.argv[1]), precalculado)