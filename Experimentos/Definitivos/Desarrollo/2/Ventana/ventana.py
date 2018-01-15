# -*- coding: utf-8 -*-
path_proyecto = '/home/guille/proyecto/proyGrado'

import sys
sys.path.append(path_proyecto)

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten
from keras.initializers import TruncatedNormal, Constant, RandomUniform
from keras.callbacks import EarlyStopping
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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def main(tarea, cantidad_iteraciones = 20, precalculado = False):

	archivo_embedding = path_proyecto + "/embedding/embedding_ordenado.txt"
	archivo_lexicon = path_proyecto + "/embedding/lexicon_total.txt"
	archivo_corpus_entrenamiento = path_proyecto + '/corpus/Ventana/Entrenamiento/' + tarea + '_training.csv'
	archivo_corpus_pruebas = path_proyecto + '/corpus/Ventana/Desarrollo/' + tarea + '_pruebas.csv'


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

	print 'Cargando casos de entrenamiento...'
	x_train, y_train = cargarCasos(archivo_corpus_entrenamiento)

	print 'Cargando casos de prueba...' 
	x_test, y_test = cargarCasos(archivo_corpus_pruebas)

	duracion_carga_casos = time.time() - inicio_carga_casos

	window_size = 11 # Cantidad de palabras en cada caso de prueba
	vector_size = 150 if precalculado else 50 # Cantidad de features a considerar por palabra
	unidades_ocultas_capa_2 = 300
	unidades_ocultas_capa_3 = len(y_train[0])

	archivo_acc = './accuracy.png'
	archivo_loss = './loss.png'

	log = 'Log de ejecucion:\n-----------------\n'
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


	# Entreno
	print 'Entrenando...'
	inicio_entrenamiento = time.time()
	early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=0, mode='auto')
	history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=cantidad_iteraciones, batch_size=100, callbacks=[early_stop], verbose=2)
	duracion_entrenamiento = time.time() - inicio_entrenamiento



	# Obtengo metricas
	print 'Obteniendo metricas...'
	inicio_metricas = time.time()

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
	(precision, recall, fscore, _) = precision_recall_fscore_support(y_true, y_pred)

	duracion_metricas = time.time() - inicio_metricas


	# list all data in history
	log += '\n\nTiempo de carga de casos de Entrenamiento/Prueba: {0} hs, {1} min, {2} s'.format(int(duracion_carga_casos/3600),int((duracion_carga_casos % 3600)/60),int((duracion_carga_casos % 3600) % 60))
	log += '\nDuracion del entrenamiento: {0} hs, {1} min, {2} s'.format(int(duracion_entrenamiento/3600),int((duracion_entrenamiento % 3600)/60),int((duracion_entrenamiento % 3600) % 60))
	log += '\nDuracion del calculo de metricas: {0} hs, {1} min, {2} s'.format(int(duracion_metricas/3600),int((duracion_metricas % 3600)/60),int((duracion_metricas % 3600) % 60))

	log += '\n\nAccuracy entrenamiento inicial: ' + str(history.history['acc'][0])
	log += '\nAccuracy entrenamiento final: ' + str(history.history['acc'][-1])
	log += '\n\nAccuracy validacion inicial: ' + str(history.history['val_acc'][0])
	log += '\nAccuracy validacion final: ' + str(history.history['val_acc'][-1])

	log += '\n\nLoss entrenamiento inicial: ' + str(history.history['loss'][0])
	log += '\nLoss entrenamiento final: ' + str(history.history['loss'][-1])
	log += '\n\nLoss validacion inicial: ' + str(history.history['val_loss'][0])
	log += '\nLoss validacion final: ' + str(history.history['val_loss'][-1])

	np.set_printoptions(threshold=10000, edgeitems=1000, linewidth=100000)

	log += '\n\nPrecision: ' + str(precision)
	log += '\nRecall: ' + str(recall)
	log += '\nMedida-F: ' + str(fscore)

	log += '\n\nMatriz de confusion:\n' + str(conf_mat)

	#print log
	open("log.txt", "w").write(BOM_UTF8 + log)

	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	#plt.show()
	plt.savefig(archivo_acc, bbox_inches='tight')

	# summarize history for loss
	plt.clf()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	#plt.show()
	plt.savefig(archivo_loss, bbox_inches='tight')


if __name__ == '__main__':
	precalculado= False if (sys.argv[3] is None or sys.argv[3] == 'a') else True
	main(str(sys.argv[1]), int(sys.argv[2]), precalculado)