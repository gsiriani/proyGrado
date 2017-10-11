# -*- coding: utf-8 -*-
path_proyecto = '/home/guille/proyecto/proyGrado'

import sys
sys.path.append(path_proyecto)

from keras.models import Model
from keras.layers import Dense, Activation, Embedding, Flatten, Conv1D, Input, Concatenate
from keras.layers.pooling import GlobalMaxPooling1D
from keras.initializers import TruncatedNormal, Constant
from keras.callbacks import EarlyStopping
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

vector_size = 50 # Cantidad de features a considerar por palabra
unidades_ocultas_capa_2 = 300
unidades_ocultas_capa_3_ner = 16 
unidades_ocultas_capa_3_chunking = 24 

archivo_embedding = path_proyecto + "/embedding/embedding_total.txt"
archivo_lexicon = path_proyecto + "/embedding/lexicon_total.txt"
archivo_corpus_entrenamiento_ner = path_proyecto + '/corpus/Oracion/Entrenamiento/ner_training.csv'
archivo_corpus_pruebas_ner = path_proyecto + '/corpus/Oracion/Pruebas/ner_pruebas.csv'
archivo_corpus_entrenamiento_chunking = path_proyecto + '/corpus/Oracion/Entrenamiento/chunking_training.csv'
archivo_corpus_pruebas_chunking = path_proyecto + '/corpus/Oracion/Pruebas/chunking_pruebas.csv'

archivo_acc_ner = './accuracy_ner.png'
archivo_loss_ner = './loss_ner.png'
archivo_acc_chunking = './accuracy_chunking.png'
archivo_loss_chunking = './loss_chunking.png'

log = 'Log de ejecucion:\n-----------------\n'
log += '\nTareas: NER y Chunking'
log += '\nModelo de red: Oracion'
log += '\nEmbedding inicial: Aleatorio'
log += '\nOptimizer: adam'


# Cargo embedding inicial
palabras = palabras_comunes(archivo_lexicon) # Indice de cada palabra en el diccionario
cant_palabras = len(palabras)  # Cantidad de palabras consideradas en el diccionario
print 'Cantidad de palabras consideradas: ' + str(cant_palabras)


# Defino las capas de la red

main_input = Input(shape=(None,), name='main_input')

aux_input_layer = Input(shape=(None,1), name='aux_input')

# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
embedding_layer = Embedding(input_dim=cant_palabras, output_dim=vector_size, embeddings_initializer='uniform',
                            trainable=True)(main_input)

concat_layer = Concatenate()([embedding_layer, aux_input_layer])

convolutive_layer = Conv1D(filters=unidades_ocultas_capa_2, kernel_size=5)(concat_layer)
#convolutive_layer = Conv1D(filters=unidades_ocultas_capa_2, kernel_size=5)(embedding_layer)

x_layer = GlobalMaxPooling1D()(convolutive_layer)

second_layer = Dense(units=unidades_ocultas_capa_2,
                     use_bias=True,
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=None),
                     bias_initializer=Constant(value=0.1))(x_layer)

y_layer = Activation("tanh")(second_layer)

salida_ner = Dense(units=unidades_ocultas_capa_3_ner,
                    activation='softmax',
                    name='salida_ner',
                    use_bias=True,
                    kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=None),
                    bias_initializer=Constant(value=0.1))(y_layer)

salida_chunking = Dense(units=unidades_ocultas_capa_3_chunking,
                    activation='softmax',
                    name='salida_chunking',
                    use_bias=True,
                    kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=None),
                    bias_initializer=Constant(value=0.1))(y_layer)


# Agrego las capas al modelo

model_ner = Model(inputs=[main_input, aux_input_layer], outputs=[salida_ner])
model_chunking = Model(inputs=[main_input, aux_input_layer], outputs=[salida_chunking])


# Compilo la red

model_ner.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ner.summary()

model_chunking.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_chunking.summary()


# Cargo Casos de entrenamiento y prueba
inicio_carga_casos = time.time()
print 'Cargando casos de entrenamiento...'

# NER
# Abro el archivo con casos de entrenamiento
x_train = []
y_train = []
with open(archivo_corpus_entrenamiento_ner, 'rb') as archivo_csv:
    lector = csv.reader(archivo_csv, delimiter=',')
    for linea in lector:
        x_train.append([int(x) for x in linea[:-unidades_ocultas_capa_3_ner]])
        y_train.append([int(x) for x in linea[-unidades_ocultas_capa_3_ner:]])

x_train_a = [l[1:] for l in x_train]
x_train_b = [ [[i-l[0]] for i in range(len(l)-1)] for l in x_train] # Matriz que almacenara distancias a la palabra a analizar
x_train_a_ner = pad_sequences(x_train_a, padding='post', value=56946)
x_train_b_ner = pad_sequences(x_train_b, padding='post', value=np.iinfo('int32').min)
y_train_ner = np.array(y_train)

# Chunking
# Abro el archivo con casos de entrenamiento
x_train = []
y_train = []
with open(archivo_corpus_entrenamiento_chunking, 'rb') as archivo_csv:
    lector = csv.reader(archivo_csv, delimiter=',')
    for linea in lector:
        x_train.append([int(x) for x in linea[:-unidades_ocultas_capa_3_chunking]])
        y_train.append([int(x) for x in linea[-unidades_ocultas_capa_3_chunking:]])

x_train_a = [l[1:] for l in x_train]
x_train_b = [ [[i-l[0]] for i in range(len(l)-1)] for l in x_train] # Matriz que almacenara distancias a la palabra a analizar
x_train_a_chunking = pad_sequences(x_train_a, padding='post', value=56946)
x_train_b_chunking = pad_sequences(x_train_b, padding='post', value=np.iinfo('int32').min)
y_train_chunking = np.array(y_train)


print 'Cargando casos de prueba...' 

# NER
# Abro el archivo con casos de prueba
x_test = []
y_test = []
with open(archivo_corpus_pruebas_ner, 'rb') as archivo_csv:
    lector = csv.reader(archivo_csv, delimiter=',')
    for linea in lector:
        x_test.append([int(x) for x in linea[:-unidades_ocultas_capa_3_ner]])
        y_test.append([int(x) for x in linea[-unidades_ocultas_capa_3_ner:]])

x_test_a = [l[1:] for l in x_test]
x_test_b = [ [[i-l[0]] for i in range(len(l)-1)] for l in x_test] # Matriz que almacenara distancias a la palabra a analizar
x_test_a_ner = pad_sequences(x_test_a, padding='post', value=56946)
x_test_b_ner = pad_sequences(x_test_b, padding='post', value=np.iinfo('int32').min)
y_test_ner = np.array(y_test)

# Chunking
# Abro el archivo con casos de prueba
x_test = []
y_test = []
with open(archivo_corpus_pruebas_chunking, 'rb') as archivo_csv:
    lector = csv.reader(archivo_csv, delimiter=',')
    for linea in lector:
        x_test.append([int(x) for x in linea[:-unidades_ocultas_capa_3_chunking]])
        y_test.append([int(x) for x in linea[-unidades_ocultas_capa_3_chunking:]])

x_test_a = [l[1:] for l in x_test]
x_test_b = [ [[i-l[0]] for i in range(len(l)-1)] for l in x_test] # Matriz que almacenara distancias a la palabra a analizar
x_test_a_chunking = pad_sequences(x_test_a, padding='post', value=56946)
x_test_b_chunking = pad_sequences(x_test_b, padding='post', value=np.iinfo('int32').min)
y_test_chunking = np.array(y_test)

duracion_carga_casos = time.time() - inicio_carga_casos


# Entreno
print 'Entrenando...'
inicio_entrenamiento = time.time()

cant_iteraciones = 5
history_ner = {'acc':[0]*cant_iteraciones, 'val_acc':[0]*cant_iteraciones, 'loss':[0]*cant_iteraciones, 'val_loss':[0]*cant_iteraciones}
history_chunking = {'acc':[0]*cant_iteraciones, 'val_acc':[0]*cant_iteraciones, 'loss':[0]*cant_iteraciones, 'val_loss':[0]*cant_iteraciones}

early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=0, mode='auto')

inicio_entrenamiento = time.time()
for i in range(cant_iteraciones):
    print 'Iteracion: ' + str(i)
    print 'NER'
    history_aux = model_ner.fit({'main_input': x_train_a_ner, 'aux_input': x_train_b_ner}, {'salida_ner': y_train_ner}, epochs=1, batch_size=100, 
                                validation_data=({'main_input': x_test_a_ner, 'aux_input': x_test_b_ner}, {'salida_ner': y_test_ner}), verbose=2)
    history_ner['acc'][i] = history_aux.history['acc'][0]
    history_ner['val_acc'][i] = history_aux.history['val_acc'][0]
    history_ner['loss'][i] = history_aux.history['loss'][0]
    history_ner['val_loss'][i] = history_aux.history['val_loss'][0]

    print 'CHUNKING'
    history_aux = model_chunking.fit({'main_input': x_train_a_chunking, 'aux_input': x_train_b_chunking}, {'salida_chunking': y_train_chunking}, epochs=1, batch_size=100, 
                                validation_data=({'main_input': x_test_a_chunking, 'aux_input': x_test_b_chunking}, {'salida_chunking': y_test_chunking}), verbose=2)
    history_chunking['acc'][i] = history_aux.history['acc'][0]
    history_chunking['val_acc'][i] = history_aux.history['val_acc'][0]
    history_chunking['loss'][i] = history_aux.history['loss'][0]
    history_chunking['val_loss'][i] = history_aux.history['val_loss'][0]

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

