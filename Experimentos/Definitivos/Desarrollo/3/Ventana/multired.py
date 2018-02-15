# -*- coding: utf-8 -*-
path_proyecto = '/home/guille/proyGrado'

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

divisor = 1000

def cargarCasos(archivo):

	# Abro el archivo
	df = pd.read_csv(archivo, sep=',', skipinitialspace=True, header=None, quoting=3)
	largo = len(df)

	# Separo features de resultados esperados
	x = np.array(df.iloc[:largo,:11])
	y = np.array(df.iloc[:largo,11:])

	return x, y

class Tarea:

	def __init__(self, nombre, cant_iteraciones):
		print 'Inicializando ' + nombre 
		self.nombre = nombre
		self.archivo_corpus_entrenamiento = path_proyecto + '/corpus/Ventana/Entrenamiento/' + nombre + '_training.csv'
		self.archivo_corpus_pruebas = path_proyecto + '/corpus/Ventana/Desarrollo/' + nombre + '_pruebas.csv'
		self.archivo_acc = './accuracy_' + nombre + '.png'
		self.archivo_loss = './loss_' + nombre + '.png'
		self.archivo_best = 'mejores_pesos_' + nombre + '.hdf5'

		print 'Cargando casos de entrenamiento de ' + nombre
		self.x_train, self.y_train = cargarCasos(self.archivo_corpus_entrenamiento)
		print 'Cargando casos de desarrollo de ' + nombre
		self.x_test, self.y_test = cargarCasos(self.archivo_corpus_pruebas)

		self.unidades_ocultas_capa_3 = len(self.y_train[0])

		self.model = Sequential()
		
		self.largo_batch = int(len(self.x_train)/divisor)

		self.history = {'acc':[], 'val_acc':[], 'loss':[], 'val_loss':[]}
		self.conf_mat = None
		self.precision = None
		self.recall = None
		self.fscore = None


	def evaluar(self):
		train = self.model.evaluate(x=self.x_train, y=self.y_train, batch_size=200, verbose=0)
		self.history['acc'].append(train[1])
		self.history['loss'].append(train[0])
		test = self.model.evaluate(x=self.x_test, y=self.y_test, batch_size=200, verbose=0)	
		self.history['val_acc'].append(test[1])
		self.history['val_loss'].append(test[0])

	def obtenerMetricas(self):
		etiquetas = range(self.unidades_ocultas_capa_3)
		predictions = self.model.predict(self.x_test, batch_size=200, verbose=0)
		y_pred = []
		for p in predictions:
		    p = p.tolist()
		    ind_max = p.index(max(p))
		    etiqueta = etiquetas[ind_max]
		    y_pred.append(etiqueta)
		y_true = []
		for p in self.y_test:
		    p = p.tolist()
		    ind_max = p.index(max(p))
		    etiqueta = etiquetas[ind_max]
		    y_true.append(etiqueta)
		self.conf_mat = confusion_matrix(y_true, y_pred, labels=etiquetas)
		(self.precision, self.recall, self.fscore, _) = precision_recall_fscore_support(y_true, y_pred)

	def graficar(self):

		# summarize history for accuracy
		plt.plot(self.history['acc'])
		plt.plot(self.history['val_acc'])
		plt.title('Accuracy ' + self.nombre.upper())
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(self.archivo_acc, bbox_inches='tight')

		# summarize history for loss
		plt.clf()
		plt.plot(self.history['loss'])
		plt.plot(self.history['val_loss'])
		plt.title('Loss ' + self.nombre.upper())
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(self.archivo_loss, bbox_inches='tight')
		plt.clf()




def main(supertag = 0, cant_iteraciones = 20, precalculado = False):

	archivo_embedding = path_proyecto + "/embedding/embedding_ordenado.txt"
	archivo_lexicon = path_proyecto + "/embedding/lexicon_total.txt"

	window_size = 11 # Cantidad de palabras en cada caso de prueba
	vector_size = 150 if precalculado else 50 # Cantidad de features a considerar por palabra
	unidades_ocultas_capa_2 = 300

	# Defino las tareas a entrenar...
	supertags = ['supertag_reducido_compactado', 'supertag_reducido', 'supertag_completo_compactado', 'supertag_completo']
	nombre_tareas = ['microchunking', 'macrochunking', 'ner', 'pos', supertags[supertag]]
	
	tareas = []
	inicio_carga_casos = time.time()
	for t in nombre_tareas:
		tareas.append(Tarea(t, cant_iteraciones))
	duracion_carga_casos = time.time() - inicio_carga_casos


	log = 'Log de ejecucion:\n-----------------\n'
	log += '\nTareas: ' + str(nombre_tareas)
	log += '\nModelo de red: Ventana'
	log += '\nEmbedding inicial: '
	if precalculado:
		log += 'Precalculado'
	else:
		log += 'Aleatorio'
	log += '\nActivacion: relu'
	log += '\nOptimizador: adam'

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
	
	# Agrego las capas a los modelos
	for t in tareas:
		t.model.add(embedding_layer)
		t.model.add(Flatten())
		t.model.add(second_layer)
		t.model.add(Activation("relu"))
		third_layer = Dense(units=t.unidades_ocultas_capa_3,
	                    use_bias=True,
	                    kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=3),
	                    bias_initializer=Constant(value=0.1))
		t.model.add(third_layer)
		t.model.add(Activation("softmax"))

		t.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		t.model.summary


	print 'Entrenando...'

	# Calculo valores iniciales
	for t in tareas:
		t.evaluar()

	# Cargo en una variable de control el valor de val_acc de la tarea de SuperTagging
	mejor_acc = tareas[-1].history['val_acc'][0]

	# Escribo en un archivo los pesos iniciales de las redes
	for t in tareas:
		t.model.save_weights(t.archivo_best)

	inicio_entrenamiento = time.time()
	for i in range(cant_iteraciones):
		print 'Iteracion: ' + str(i+1)
		for j in range(divisor):
			#print_progress(j, divisor)
			for t in tareas:
				_ = t.model.fit(t.x_train[t.largo_batch*j: t.largo_batch*(j+1)], t.y_train[t.largo_batch*j: t.largo_batch*(j+1)],
								epochs=1, batch_size=100, verbose=0)

		#print_progress(divisor, divisor)
		print '\n'
		for t in tareas:
			t.evaluar()

		# Actualizo pesos optimos
		if tareas[-1].history['val_acc'][-1] > mejor_acc:
			mejor_acc = tareas[-1].history['val_acc'][-1]
			# Actualizo los pesos de las redes
			for t in tareas:
				t.model.save_weights(t.archivo_best)

	duracion_entrenamiento = time.time() - inicio_entrenamiento

	

	# Obtengo metricas
	print 'Obteniendo metricas...'
	inicio_metricas = time.time()

	for t in tareas:
		t.model.load_weights(t.archivo_best)
		t.obtenerMetricas()
		t.evaluar()

	duracion_metricas = time.time() - inicio_metricas

	# Escribo en log
	log += '\n\nTiempo de carga de casos de Entrenamiento/Prueba: {0} hs, {1} min, {2} s'.format(int(duracion_carga_casos/3600),int((duracion_carga_casos % 3600)/60),int((duracion_carga_casos % 3600) % 60))
	log += '\nDuracion del entrenamiento: {0} hs, {1} min, {2} s'.format(int(duracion_entrenamiento/3600),int((duracion_entrenamiento % 3600)/60),int((duracion_entrenamiento % 3600) % 60))

	np.set_printoptions(threshold=10000, edgeitems=1000, linewidth=100000)

	for t in tareas:		
		log += '\n\n' + t.nombre.upper() + '\n--------'
		log += '\n\nAccuracy entrenamiento inicial: ' + str(t.history['acc'][0])
		log += '\nAccuracy entrenamiento final: ' + str(t.history['acc'][-1])
		log += '\n\nAccuracy validacion inicial: ' + str(t.history['val_acc'][0])
		log += '\nAccuracy validacion final: ' + str(t.history['val_acc'][-1])

		log += '\n\nLoss entrenamiento inicial: ' + str(t.history['loss'][0])
		log += '\nLoss entrenamiento final: ' + str(t.history['loss'][-1])
		log += '\n\nLoss validacion inicial: ' + str(t.history['val_loss'][0])
		log += '\nLoss validacion final: ' + str(t.history['val_loss'][-1])

		log += '\n\nPrecision: ' + str(t.precision)
		log += '\nRecall: ' + str(t.recall)
		log += '\nMedida-F: ' + str(t.fscore)

		log += '\n\nMatriz de confusion:\n' + str(t.conf_mat)


	#print log
	open("log.txt", "w").write(BOM_UTF8 + log)

	for t in tareas:
		t.graficar()


if __name__ == '__main__':
	precalculado = False if (sys.argv[3] is None or sys.argv[3] == 'a') else True
	main(int(sys.argv[1]), int(sys.argv[2]), precalculado)