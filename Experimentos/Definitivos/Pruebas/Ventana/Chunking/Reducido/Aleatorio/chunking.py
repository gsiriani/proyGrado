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
vector_size = 50 # Cantidad de features a considerar por palabra
unidades_ocultas_capa_2 = 300
unidades_ocultas_capa_3 = 9 # SE MODIFICA PARA CADA PROBLEMA A RESOLVER

cantidad_iteraciones = 200

archivo_embedding = path_proyecto + "/embedding/embedding_total.txt"
archivo_lexicon = path_proyecto + "/embedding/lexicon_total.txt"
archivo_corpus_entrenamiento = path_proyecto + '/corpus/Ventana/Entrenamiento/chunking_reducido_training.csv'
archivo_corpus_pruebas = path_proyecto + '/corpus/Ventana/Pruebas/chunking_reducido_pruebas.csv'

archivo_acc = './accuracy.png'
archivo_loss = './loss.png'

log = 'Log de ejecucion:\n-----------------\n'
log += '\nTarea: Chunking Reducido'
log += '\nModelo de red: Ventana'
log += '\nEmbedding inicial: Aleatorio'
log += '\nOptimizer: adam'
log += '\nActivacion: relu'

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
inicio_carga_casos = time.time()
print 'Cargando casos de entrenamiento...'

# Abro el archivo con casos de entrenamiento
df = pd.read_csv(archivo_corpus_entrenamiento, sep=',', skipinitialspace=True, header=None, quoting=3)
largo = len(df)

# Separo features de resultados esperados
x_train = np.array(df.iloc[:largo,:11])
y_train = np.array(df.iloc[:largo,11:])


print 'Cargando casos de prueba...' 

# Abro el archivo con casos de prueba
df = pd.read_csv(archivo_corpus_pruebas, sep=',', skipinitialspace=True, header=None, quoting=3)
largo = len(df)
cantidad_casos_test = largo

# Separo features de resultados esperados
x_test = np.array(df.iloc[:largo,:11])
y_test = np.array(df.iloc[:largo,11:])

duracion_carga_casos = time.time() - inicio_carga_casos


print 'Entrenando...'
inicio_entrenamiento = time.time()
early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=6, verbose=0, mode='auto')
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=cantidad_iteraciones, batch_size=100, 
					callbacks=[early_stop], verbose=2)
duracion_entrenamiento = time.time() - inicio_entrenamiento




# list all data in history
log += '\n\nTiempo de carga de casos de Entrenamiento/Prueba: {0} hs, {1} min, {2} s'.format(int(duracion_carga_casos/3600),int((duracion_carga_casos % 3600)/60),int((duracion_carga_casos % 3600) % 60))
log += '\nDuracion del entrenamiento: {0} hs, {1} min, {2} s'.format(int(duracion_entrenamiento/3600),int((duracion_entrenamiento % 3600)/60),int((duracion_entrenamiento % 3600) % 60))

log += '\n\nAccuracy entrenamiento inicial: ' + str(history.history['acc'][0])
log += '\nAccuracy entrenamiento final: ' + str(history.history['acc'][-1])
log += '\n\nAccuracy validacion inicial: ' + str(history.history['val_acc'][0])
log += '\nAccuracy validacion final: ' + str(history.history['val_acc'][-1])

log += '\n\nLoss entrenamiento inicial: ' + str(history.history['loss'][0])
log += '\nLoss entrenamiento final: ' + str(history.history['loss'][-1])
log += '\n\nLoss validacion inicial: ' + str(history.history['val_loss'][0])
log += '\nLoss validacion final: ' + str(history.history['val_loss'][-1])

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
