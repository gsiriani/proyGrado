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

window_size = 11 # Cantidad de palabras en cada caso de prueba
vector_size = 150 # Cantidad de features a considerar por palabra
unidades_ocultas_capa_2 = 300
unidades_ocultas_capa_3_ner = 17 
unidades_ocultas_capa_3_chunking = 25 
unidades_ocultas_capa_3_pos = 12
unidades_ocultas_capa_3_str = 643 
unidades_ocultas_capa_3_sta = 947 

cant_iteraciones = 20

archivo_embedding = path_proyecto + "/embedding/embedding_ordenado.txt"
archivo_lexicon = path_proyecto + "/embedding/lexicon_total.txt"
archivo_corpus_entrenamiento_ner = path_proyecto + '/corpus/Ventana/Entrenamiento/ner_training.csv'
archivo_corpus_pruebas_ner = path_proyecto + '/corpus/Ventana/Pruebas/ner_pruebas.csv'
archivo_corpus_entrenamiento_chunking = path_proyecto + '/corpus/Ventana/Entrenamiento/chunking_training.csv'
archivo_corpus_pruebas_chunking = path_proyecto + '/corpus/Ventana/Pruebas/chunking_pruebas.csv'
archivo_corpus_entrenamiento_pos = path_proyecto + '/corpus/Ventana/Entrenamiento/pos_simple_training.csv'
archivo_corpus_pruebas_pos = path_proyecto + '/corpus/Ventana/Pruebas/pos_simple_pruebas.csv'
archivo_corpus_entrenamiento_str = path_proyecto + '/corpus/Ventana/Entrenamiento/supertag_reducido_training.csv'
archivo_corpus_pruebas_str = path_proyecto + '/corpus/Ventana/Pruebas/supertag_reducido_pruebas.csv'

archivo_acc_ner = './accuracy_ner.png'
archivo_loss_ner = './loss_ner.png'
archivo_acc_chunking = './accuracy_chunking.png'
archivo_loss_chunking = './loss_chunking.png'
archivo_acc_pos = './accuracy_pos.png'
archivo_loss_pos = './loss_pos.png'
archivo_acc_str = './accuracy_st_reducido.png'
archivo_loss_str = './loss_st_reducido.png'

log = 'Log de ejecucion:\n-----------------\n'
log += '\nTareas: NER, Chunking, POS Simple y Super Tagging Reducido'
log += '\nEmbedding inicial: Aleatorio'
log += '\nActivacion: relu'
log += '\nOptimizador: adam'

print 'Cargando embedding inicial...'
# Cargo embedding inicial
palabras = palabras_comunes(archivo_lexicon) # Indice de cada palabra en el diccionario

cant_palabras = len(palabras)  # Cantidad de palabras consideradas en el diccionario

# Defino las capas de la red

# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
embedding_layer = Embedding(input_dim=cant_palabras, output_dim=vector_size,
                            embeddings_initializer=RandomUniform(minval=-0.05, maxval=0.05, seed=1),
                            input_length=window_size, trainable=True)

second_layer = Dense(units=unidades_ocultas_capa_2,
                     use_bias=True,
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=2),
                     bias_initializer=Constant(value=0.1))

third_layer_ner = Dense(units=unidades_ocultas_capa_3_ner,
                    use_bias=True,
                    kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=3),
                    bias_initializer=Constant(value=0.1))

third_layer_chunking = Dense(units=unidades_ocultas_capa_3_chunking,
                    use_bias=True,
                    kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=3),
                    bias_initializer=Constant(value=0.1))

third_layer_pos = Dense(units=unidades_ocultas_capa_3_pos,
                    use_bias=True,
                    kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=3),
                    bias_initializer=Constant(value=0.1))

third_layer_str = Dense(units=unidades_ocultas_capa_3_str,
                    use_bias=True,
                    kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=3),
                    bias_initializer=Constant(value=0.1))



# Agrego las capas a los modelos

# NER
model_ner = Sequential()

model_ner.add(embedding_layer)
model_ner.add(Flatten())
model_ner.add(second_layer)
model_ner.add(Activation("relu"))
model_ner.add(third_layer_ner)
model_ner.add(Activation("softmax"))


# CHUNKING
model_chunking = Sequential()

model_chunking.add(embedding_layer)
model_chunking.add(Flatten())
model_chunking.add(second_layer)
model_chunking.add(Activation("relu"))
model_chunking.add(third_layer_chunking)
model_chunking.add(Activation("softmax"))

# POS
model_pos = Sequential()

model_pos.add(embedding_layer)
model_pos.add(Flatten())
model_pos.add(second_layer)
model_pos.add(Activation("relu"))
model_pos.add(third_layer_pos)
model_pos.add(Activation("softmax"))

# ST Reducido
model_str = Sequential()

model_str.add(embedding_layer)
model_str.add(Flatten())
model_str.add(second_layer)
model_str.add(Activation("relu"))
model_str.add(third_layer_str)
model_str.add(Activation("softmax"))



# Compilo las redes

model_ner.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ner.summary()
model_chunking.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_chunking.summary()
model_pos.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_pos.summary()
model_str.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_str.summary()


# Entreno
inicio_carga_casos = time.time()
print 'Cargando casos de entrenamiento...'

# NER

# Abro el archivo con casos de entrenamiento
df = pd.read_csv(archivo_corpus_entrenamiento_ner, sep=',', skipinitialspace=True, header=None, quoting=3)
largo = len(df)

# Separo features de resultados esperados
x_train_ner = np.array(df.iloc[:largo,:11])
y_train_ner = np.array(df.iloc[:largo,11:])

# CHUNKING

# Abro el archivo con casos de entrenamiento
df = pd.read_csv(archivo_corpus_entrenamiento_chunking, sep=',', skipinitialspace=True, header=None, quoting=3)
largo = len(df)

# Separo features de resultados esperados
x_train_chunking = np.array(df.iloc[:largo,:11])
y_train_chunking = np.array(df.iloc[:largo,11:])

# POS

# Abro el archivo con casos de entrenamiento
df = pd.read_csv(archivo_corpus_entrenamiento_pos, sep=',', skipinitialspace=True, header=None, quoting=3)
largo = len(df)

# Separo features de resultados esperados
x_train_pos = np.array(df.iloc[:largo,:11])
y_train_pos = np.array(df.iloc[:largo,11:])

# ST Reducido

# Abro el archivo con casos de entrenamiento
df = pd.read_csv(archivo_corpus_entrenamiento_str, sep=',', skipinitialspace=True, header=None, quoting=3)
largo = len(df)

# Separo features de resultados esperados
x_train_str = np.array(df.iloc[:largo,:11])
y_train_str = np.array(df.iloc[:largo,11:])


print 'Cargando casos de prueba...' 

# NER

# Abro el archivo con casos de prueba
df = pd.read_csv(archivo_corpus_pruebas_ner, sep=',', skipinitialspace=True, header=None, quoting=3)
largo = len(df)

# Separo features de resultados esperados
x_test_ner = np.array(df.iloc[:largo,:11])
y_test_ner = np.array(df.iloc[:largo,11:])

# CHUNKING

# Abro el archivo con casos de prueba
df = pd.read_csv(archivo_corpus_pruebas_chunking, sep=',', skipinitialspace=True, header=None, quoting=3)
largo = len(df)

# Separo features de resultados esperados
x_test_chunking = np.array(df.iloc[:largo,:11])
y_test_chunking = np.array(df.iloc[:largo,11:])

# POS

# Abro el archivo con casos de prueba
df = pd.read_csv(archivo_corpus_pruebas_pos, sep=',', skipinitialspace=True, header=None, quoting=3)
largo = len(df)

# Separo features de resultados esperados
x_test_pos = np.array(df.iloc[:largo,:11])
y_test_pos = np.array(df.iloc[:largo,11:])

# ST Reducido

# Abro el archivo con casos de prueba
df = pd.read_csv(archivo_corpus_pruebas_str, sep=',', skipinitialspace=True, header=None, quoting=3)
largo = len(df)

# Separo features de resultados esperados
x_test_str = np.array(df.iloc[:largo,:11])
y_test_str = np.array(df.iloc[:largo,11:])


duracion_carga_casos = time.time() - inicio_carga_casos

history_ner = {'acc':[0]*cant_iteraciones, 'val_acc':[0]*cant_iteraciones, 'loss':[0]*cant_iteraciones, 'val_loss':[0]*cant_iteraciones}
history_chunking = {'acc':[0]*cant_iteraciones, 'val_acc':[0]*cant_iteraciones, 'loss':[0]*cant_iteraciones, 'val_loss':[0]*cant_iteraciones}
history_pos = {'acc':[0]*cant_iteraciones, 'val_acc':[0]*cant_iteraciones, 'loss':[0]*cant_iteraciones, 'val_loss':[0]*cant_iteraciones}
history_str = {'acc':[0]*cant_iteraciones, 'val_acc':[0]*cant_iteraciones, 'loss':[0]*cant_iteraciones, 'val_loss':[0]*cant_iteraciones}

print 'Entrenando...'
inicio_entrenamiento = time.time()
for i in range(cant_iteraciones):
	print 'Iteracion: ' + str(i)
	print 'NER'
	history_aux = model_ner.fit(x_train_ner, y_train_ner, validation_data=(x_test_ner, y_test_ner), epochs=1, batch_size=100, verbose=2)
	history_ner['acc'][i] = history_aux.history['acc'][0]
	history_ner['val_acc'][i] = history_aux.history['val_acc'][0]
	history_ner['loss'][i] = history_aux.history['loss'][0]
	history_ner['val_loss'][i] = history_aux.history['val_loss'][0]

	print 'CHUNKING'
	history_aux = model_chunking.fit(x_train_chunking, y_train_chunking, validation_data=(x_test_chunking, y_test_chunking), epochs=1, batch_size=100, verbose=2)
	history_chunking['acc'][i] = history_aux.history['acc'][0]
	history_chunking['val_acc'][i] = history_aux.history['val_acc'][0]
	history_chunking['loss'][i] = history_aux.history['loss'][0]
	history_chunking['val_loss'][i] = history_aux.history['val_loss'][0]

	print 'POS'
	history_aux = model_pos.fit(x_train_pos, y_train_pos, validation_data=(x_test_pos, y_test_pos), epochs=1, batch_size=100, verbose=2)
	history_pos['acc'][i] = history_aux.history['acc'][0]
	history_pos['val_acc'][i] = history_aux.history['val_acc'][0]
	history_pos['loss'][i] = history_aux.history['loss'][0]
	history_pos['val_loss'][i] = history_aux.history['val_loss'][0]

	print 'ST Reducido'
	history_aux = model_str.fit(x_train_str, y_train_str, validation_data=(x_test_str, y_test_str), epochs=1, batch_size=100, verbose=2)
	history_str['acc'][i] = history_aux.history['acc'][0]
	history_str['val_acc'][i] = history_aux.history['val_acc'][0]
	history_str['loss'][i] = history_aux.history['loss'][0]
	history_str['val_loss'][i] = history_aux.history['val_loss'][0]

duracion_entrenamiento = time.time() - inicio_entrenamiento

# list all data in history
log += '\n\nTiempo de carga de casos de Entrenamiento/Prueba: {0} hs, {1} min, {2} s'.format(int(duracion_carga_casos/3600),int((duracion_carga_casos % 3600)/60),int((duracion_carga_casos % 3600) % 60))
log += '\nDuracion del entrenamiento: {0} hs, {1} min, {2} s'.format(int(duracion_entrenamiento/3600),int((duracion_entrenamiento % 3600)/60),int((duracion_entrenamiento % 3600) % 60))

log += '\n\nNER\n--------'
log += '\n\nAccuracy entrenamiento inicial: ' + str(history_ner['acc'][0])
log += '\nAccuracy entrenamiento final: ' + str(history_ner['acc'][-1])
log += '\n\nAccuracy validacion inicial: ' + str(history_ner['val_acc'][0])
log += '\nAccuracy validacion final: ' + str(history_ner['val_acc'][-1])

log += '\n\nLoss entrenamiento inicial: ' + str(history_ner['loss'][0])
log += '\nLoss entrenamiento final: ' + str(history_ner['loss'][-1])
log += '\n\nLoss validacion inicial: ' + str(history_ner['val_loss'][0])
log += '\nLoss validacion final: ' + str(history_ner['val_loss'][-1])

log += '\n\nCHUNKING\n--------'
log += '\n\nAccuracy entrenamiento inicial: ' + str(history_chunking['acc'][0])
log += '\nAccuracy entrenamiento final: ' + str(history_chunking['acc'][-1])
log += '\n\nAccuracy validacion inicial: ' + str(history_chunking['val_acc'][0])
log += '\nAccuracy validacion final: ' + str(history_chunking['val_acc'][-1])

log += '\n\nLoss entrenamiento inicial: ' + str(history_chunking['loss'][0])
log += '\nLoss entrenamiento final: ' + str(history_chunking['loss'][-1])
log += '\n\nLoss validacion inicial: ' + str(history_chunking['val_loss'][0])
log += '\nLoss validacion final: ' + str(history_chunking['val_loss'][-1])

log += '\n\nPOS\n--------'
log += '\n\nAccuracy entrenamiento inicial: ' + str(history_pos['acc'][0])
log += '\nAccuracy entrenamiento final: ' + str(history_pos['acc'][-1])
log += '\n\nAccuracy validacion inicial: ' + str(history_pos['val_acc'][0])
log += '\nAccuracy validacion final: ' + str(history_pos['val_acc'][-1])

log += '\n\nLoss entrenamiento inicial: ' + str(history_pos['loss'][0])
log += '\nLoss entrenamiento final: ' + str(history_pos['loss'][-1])
log += '\n\nLoss validacion inicial: ' + str(history_pos['val_loss'][0])
log += '\nLoss validacion final: ' + str(history_pos['val_loss'][-1])

log += '\n\nSuperTagging Reducido\n--------'
log += '\n\nAccuracy entrenamiento inicial: ' + str(history_str['acc'][0])
log += '\nAccuracy entrenamiento final: ' + str(history_str['acc'][-1])
log += '\n\nAccuracy validacion inicial: ' + str(history_str['val_acc'][0])
log += '\nAccuracy validacion final: ' + str(history_str['val_acc'][-1])

log += '\n\nLoss entrenamiento inicial: ' + str(history_str['loss'][0])
log += '\nLoss entrenamiento final: ' + str(history_str['loss'][-1])
log += '\n\nLoss validacion inicial: ' + str(history_str['val_loss'][0])
log += '\nLoss validacion final: ' + str(history_str['val_loss'][-1])


#print log
open("log.txt", "w").write(BOM_UTF8 + log)

# summarize history for accuracy
plt.plot(history_ner['acc'])
plt.plot(history_ner['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig(archivo_acc_ner, bbox_inches='tight')

# summarize history for loss
plt.clf()
plt.plot(history_ner['loss'])
plt.plot(history_ner['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig(archivo_loss_ner, bbox_inches='tight')
plt.clf()

# summarize history for accuracy
plt.plot(history_chunking['acc'])
plt.plot(history_chunking['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig(archivo_acc_chunking, bbox_inches='tight')

# summarize history for loss
plt.clf()
plt.plot(history_chunking['loss'])
plt.plot(history_chunking['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig(archivo_loss_chunking, bbox_inches='tight')
plt.clf()

# summarize history for accuracy
plt.plot(history_pos['acc'])
plt.plot(history_pos['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig(archivo_acc_pos, bbox_inches='tight')

# summarize history for loss
plt.clf()
plt.plot(history_pos['loss'])
plt.plot(history_pos['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig(archivo_loss_pos, bbox_inches='tight')
plt.clf()

# summarize history for accuracy
plt.plot(history_str['acc'])
plt.plot(history_str['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig(archivo_acc_str, bbox_inches='tight')

# summarize history for loss
plt.clf()
plt.plot(history_str['loss'])
plt.plot(history_str['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig(archivo_loss_str, bbox_inches='tight')
plt.clf()
