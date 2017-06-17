# -*- coding: utf-8 -*-
path_proyecto = '/home/guille/proyGrado'

import sys
sys.path.append(path_proyecto)

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten
from keras.initializers import TruncatedNormal, Constant
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
unidades_ocultas_capa_3 = 24 # SE MODIFICA PARA CADA PROBLEMA A RESOLVER

archivo_embedding = path_proyecto + "/embedding/embedding_total.txt"
archivo_lexicon = path_proyecto + "/embedding/lexicon_total.txt"
archivo_corpus_entrenamiento = path_proyecto + '/corpus/Ventana_indizada/Entrenamiento/chunking_training.csv'
archivo_corpus_pruebas = path_proyecto + '/corpus/Ventana_indizada/Pruebas/chunking_pruebas.csv'

archivo_acc = './accuracy.png'
archivo_loss = './loss.png'

log = 'Log de ejecucion:\n-----------------\n'
log += '\nTarea: Chunking'
log += '\nModelo de red: Ventana'
log += '\nEmbedding inicial: Precalculado'
log += '\nIOBES: Unido'
log += '\nActivacion: tanh'
log += '\nLearning Rate / Momentum: 0.01 / 0'
log += '\nOUT tag: NO'

print 'Cargando embedding inicial...'
# Cargo embedding inicial
palabras = palabras_comunes(archivo_lexicon) # Indice de cada palabra en el diccionario

cant_palabras = len(palabras)  # Cantidad de palabras consideradas en el diccionario
f = 0
embedding_inicial= [[0] * vector_size] * len(palabras)
for l in open(archivo_embedding):
    print_progress(f, cant_palabras, prefix = 'Progreso:', suffix = 'Completado', bar_length = 50)
    f += 1
    ind_palabra = palabras.obtener_indice(l.split()[0])
    embedding_inicial[ind_palabra] = list([float(x) for x in l.split()[1:]])
    #embedding_inicial.append([float(x) for x in l.split()[1:]])


# Agregamos embedding para 11 signos de puntuacion y PUNCT inicializado como el mismo embedding que ':'
signos_puntuacion = ['.', ',', ';', '(', ')', '¿', '?', '¡', '!', '"', "'", 'PUNCT'] 
indice_punct_base = palabras.obtener_indice(':')
for s in signos_puntuacion:
    ind_palabra = palabras.obtener_indice(s)
    embedding_inicial[ind_palabra] = list(embedding_inicial[indice_punct_base])

# Agregamos embedding para NUM, DATE, OUT y UNK
palabras_extra = ['NUM', 'DATE', 'OUT', 'UNK']
for p in palabras_extra:
    features_aux = []
    for _ in range(vector_size):
        features_aux.append(uniform(-1,1))    
    ind_palabra = palabras.obtener_indice(p)
    embedding_inicial[ind_palabra] = list(features_aux)


embedding_inicial = np.array(embedding_inicial)

cant_palabras = len(embedding_inicial)
print 'Cantidad de palabras consideradas: ' + str(cant_palabras)

print 'Dimensiones del embedding: ' + str(embedding_inicial.shape)
 

# Defino las capas de la red

# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
embedding_layer = Embedding(input_dim=cant_palabras, output_dim=vector_size, weights=[embedding_inicial],
                            input_length=window_size, trainable=True)

second_layer = Dense(units=unidades_ocultas_capa_2,
                     use_bias=True,
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=None),
                     bias_initializer=Constant(value=0.1))

third_layer = Dense(units=unidades_ocultas_capa_3,
                    use_bias=True,
                    kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=None),
                    bias_initializer=Constant(value=0.1))


# Agrego las capas al modelo

model = Sequential()

model.add(embedding_layer)
model.add(Flatten())
model.add(second_layer)
model.add(Activation("tanh"))
model.add(third_layer)
# model.add(Activation("softmax"))


# Compilo la red
sgd = optimizers.SGD(lr=0.01, momentum=0)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
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

# Separo features de resultados esperados
x_test = np.array(df.iloc[:largo,:11])
y_test = np.array(df.iloc[:largo,11:])

duracion_carga_casos = time.time() - inicio_carga_casos


print 'Entrenando...'
inicio_entrenamiento = time.time()
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=200, batch_size=100, verbose=2)
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
