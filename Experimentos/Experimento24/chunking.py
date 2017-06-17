# -*- coding: utf-8 -*-
path_proyecto = '/home/guille/proyGrado'

import sys
sys.path.append(path_proyecto)

from keras.models import Model
from keras.layers import Dense, Activation, Embedding, Flatten, Input
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
neuronas_salida_main = 6 # SE MODIFICA PARA CADA PROBLEMA A RESOLVER
neuronas_salida_iobes = 4

archivo_embedding = path_proyecto + "/embedding/embedding_total.txt"
archivo_lexicon = path_proyecto + "/embedding/lexicon_total.txt"
archivo_corpus_entrenamiento = path_proyecto + '/corpus/Ventana_indizada/Entrenamiento/chunking_training_iobes_separado.csv'
archivo_corpus_pruebas = path_proyecto + '/corpus/Ventana_indizada/Pruebas/chunking_pruebas_iobes_separado.csv'

archivo_acc_main = './accuracy_main.png'
archivo_acc_iobes = './accuracy_iobes.png'

log = 'Log de ejecucion:\n-----------------\n'
log += '\nTarea: Chunking'
log += '\nModelo de red: Ventana'
log += '\nEmbedding inicial: Aleatorio'
log += '\nIOBES: Separado'

print 'Cargando embedding inicial...'
# Cargo embedding inicial
palabras = palabras_comunes(archivo_lexicon) # Indice de cada palabra en el diccionario

cant_palabras = len(palabras)  # Cantidad de palabras consideradas en el diccionario

# Defino las capas de la red

# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
input_layer = Input(shape=(window_size,))

embedding_layer = Embedding(input_dim=cant_palabras, output_dim=vector_size,
                            #input_length=window_size, 
                            trainable=True)(input_layer)

flatten_layer = Flatten()(embedding_layer)

second_layer = Dense(units=unidades_ocultas_capa_2,
                     use_bias=True,
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=None),
                     bias_initializer=Constant(value=0.1))(flatten_layer)

tanh_layer = Activation("tanh")(second_layer)

salida_main = Dense(units=neuronas_salida_main,
                    use_bias=True,
                    kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=None),
                    bias_initializer=Constant(value=0.1),
                    name='salida_main')(tanh_layer)

salida_iobes = Dense(units=neuronas_salida_iobes,
                    use_bias=True,
                    kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=None),
                    bias_initializer=Constant(value=0.1),
                    name='salida_iobes')(tanh_layer)


# Agrego las capas al modelo

model = Model(inputs=[input_layer], outputs=[salida_main, salida_iobes])


# Compilo la red
sgd = optimizers.SGD(lr=0.1, momentum=0.02)
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
y_train_main = np.array(df.iloc[:largo,11:11 + neuronas_salida_main])
y_train_iobes = np.array(df.iloc[:largo,11 + neuronas_salida_main:])


print 'Cargando casos de prueba...' 

# Abro el archivo con casos de prueba
df = pd.read_csv(archivo_corpus_pruebas, sep=',', skipinitialspace=True, header=None, quoting=3)
largo = len(df)

# Separo features de resultados esperados
x_test = np.array(df.iloc[:largo,:11])
y_test_main = np.array(df.iloc[:largo,11:11 + neuronas_salida_main])
y_test_iobes = np.array(df.iloc[:largo,11 + neuronas_salida_main:])

duracion_carga_casos = time.time() - inicio_carga_casos


print 'Entrenando...'
inicio_entrenamiento = time.time()
history = model.fit(x_train, 
                    {'salida_main': y_train_main, 'salida_iobes': y_train_iobes}, 
                    validation_data=(x_test, {'salida_main': y_test_main, 'salida_iobes': y_test_iobes}), 
                    epochs=50, 
                    batch_size=100, 
                    verbose=2)
duracion_entrenamiento = time.time() - inicio_entrenamiento

# list all data in history
log += '\n\nTiempo de carga de casos de Entrenamiento/Prueba: {0} hs, {1} min, {2} s'.format(int(duracion_carga_casos/3600),int((duracion_carga_casos % 3600)/60),int((duracion_carga_casos % 3600) % 60))
log += '\nDuracion del entrenamiento: {0} hs, {1} min, {2} s'.format(int(duracion_entrenamiento/3600),int((duracion_entrenamiento % 3600)/60),int((duracion_entrenamiento % 3600) % 60))

log += '\n\nAccuracy main entrenamiento inicial: ' + str(history.history['salida_main_acc'][0])
log += '\nAccuracy main entrenamiento final: ' + str(history.history['salida_main_acc'][-1])
log += '\n\nAccuracy main validacion inicial: ' + str(history.history['val_salida_main_acc'][0])
log += '\nAccuracy main validacion final: ' + str(history.history['val_salida_main_acc'][-1])

log += '\n\nAccuracy iobes entrenamiento inicial: ' + str(history.history['salida_iobes_acc'][0])
log += '\nAccuracy iobes entrenamiento final: ' + str(history.history['salida_iobes_acc'][-1])
log += '\n\nAccuracy iobes validacion inicial: ' + str(history.history['val_salida_iobes_acc'][0])
log += '\nAccuracy iobes validacion final: ' + str(history.history['val_salida_iobes_acc'][-1])

#print log
open("log.txt", "w").write(BOM_UTF8 + log)

# summarize history for accuracy main
plt.plot(history.history['salida_main_acc'])
plt.plot(history.history['val_salida_main_acc'])
plt.title('model accuracy main')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig(archivo_acc_main, bbox_inches='tight')

plt.clf()

# summarize history for accuracy main
plt.plot(history.history['salida_iobes_acc'])
plt.plot(history.history['val_salida_iobes_acc'])
plt.title('model accuracy iobes')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig(archivo_acc_iobes, bbox_inches='tight')
