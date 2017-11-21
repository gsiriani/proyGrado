# -*- coding: utf-8 -*-
path_proyecto = '/home/guille/proyecto/proyGrado'

import sys
sys.path.append(path_proyecto)

from keras.models import Model
from keras.layers import Dense, Activation, Embedding, Flatten, Conv1D, Input, Concatenate
from keras.layers.pooling import GlobalMaxPooling1D
from keras.initializers import TruncatedNormal, Constant, RandomUniform
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
unidades_ocultas_capa_3_ner = 17 
unidades_ocultas_capa_3_chunking = 25 
unidades_ocultas_capa_3_pos = 12
unidades_ocultas_capa_3_st = 643 # Reducido
#unidades_ocultas_capa_3_st = 947 # Completo
unidades_ocultas_capa_3_srl = 33
largo_oracion = 50

cant_iteraciones = 200

archivo_embedding = path_proyecto + "/embedding/lexicon_total.txt"

archivo_corpus_entrenamiento_ner = path_proyecto + '/corpus/Sentencia_truncada/Entrenamiento/ner_training.csv'
archivo_corpus_pruebas_ner = path_proyecto + '/corpus/Sentencia_truncada/Pruebas/ner_pruebas.csv'
archivo_corpus_entrenamiento_chunking = path_proyecto + '/corpus/Sentencia_truncada/Entrenamiento/chunking_training.csv'
archivo_corpus_pruebas_chunking = path_proyecto + '/corpus/Sentencia_truncada/Pruebas/chunking_pruebas.csv'
archivo_corpus_entrenamiento_pos = path_proyecto + '/corpus/Sentencia_truncada/Entrenamiento/pos_simple_training.csv'
archivo_corpus_pruebas_pos = path_proyecto + '/corpus/Sentencia_truncada/Pruebas/pos_simple_pruebas.csv'
archivo_corpus_entrenamiento_st = path_proyecto + '/corpus/Sentencia_truncada/Entrenamiento/supertag_reducido_training.csv'
archivo_corpus_pruebas_st = path_proyecto + '/corpus/Sentencia_truncada/Pruebas/supertag_reducido_pruebas.csv'
#archivo_corpus_entrenamiento_st = path_proyecto + '/corpus/Sentencia_truncada/Entrenamiento/supertag_completo_training.csv'
#archivo_corpus_pruebas_st = path_proyecto + '/corpus/Sentencia_truncada/Pruebas/supertag_completo_pruebas.csv'
archivo_corpus_entrenamiento_srl = path_proyecto + '/corpus/Sentencia_truncada/Entrenamiento/srl_simple_training.csv'
archivo_corpus_pruebas_srl = path_proyecto + '/corpus/Sentencia_truncada/Pruebas/srl_simple_pruebas.csv'

archivo_acc_ner = './accuracy_ner.png'
archivo_loss_ner = './loss_ner.png'
archivo_acc_chunking = './accuracy_chunking.png'
archivo_loss_chunking = './loss_chunking.png'
archivo_acc_pos = './accuracy_pos.png'
archivo_loss_pos = './loss_pos.png'
archivo_acc_st = './accuracy_supertag.png'
archivo_loss_st = './loss_supertag.png'
archivo_acc_srl = './accuracy_srl.png'
archivo_loss_srl = './loss_srl.png'

log = 'Log de ejecucion:\n-----------------\n'
log += '\nTareas: NER, Chunking, POS Simple, SRL y Super Tagging Reducido'
log += '\nModelo de red: Oracion'
log += '\nEmbedding inicial: Aleatorio'
log += '\nOptimizer: adam'


# Cargo embedding inicial
palabras = palabras_comunes(archivo_embedding) # Indice de cada palabra en el diccionario
indice_OUT = palabras.obtener_indice("OUT")
cant_palabras = len(palabras)  # Cantidad de palabras consideradas en el diccionario
print 'Cantidad de palabras consideradas: ' + str(cant_palabras)


# Defino las capas de la red

main_input = Input(shape=(None,), name='main_input')

aux_input_layer = Input(shape=(None,2), name='aux_input')

# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
embedding_layer = Embedding(input_dim=cant_palabras, output_dim=vector_size,
                            embeddings_initializer=RandomUniform(minval=-0.05, maxval=0.05, seed=1),
                            trainable=True)(main_input)                            

concat_layer = Concatenate()([embedding_layer, aux_input_layer])

convolutive_layer = Conv1D(filters=unidades_ocultas_capa_2, kernel_size=5)(concat_layer)

x_layer = GlobalMaxPooling1D()(convolutive_layer)

second_layer = Dense(units=unidades_ocultas_capa_2,
                     use_bias=True,
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=2),
                     bias_initializer=Constant(value=0.1))(x_layer)

y_layer = Activation("tanh")(second_layer)


third_layer_ner = Dense(units=unidades_ocultas_capa_3_ner,
                    use_bias=True,
                    kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=3),
                    bias_initializer=Constant(value=0.1))(y_layer)

third_layer_chunking = Dense(units=unidades_ocultas_capa_3_chunking,
                    use_bias=True,
                    kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=3),
                    bias_initializer=Constant(value=0.1))(y_layer)

third_layer_pos = Dense(units=unidades_ocultas_capa_3_pos,
                    use_bias=True,
                    kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=3),
                    bias_initializer=Constant(value=0.1))(y_layer)

third_layer_st = Dense(units=unidades_ocultas_capa_3_st,
                    use_bias=True,
                    kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=3),
                    bias_initializer=Constant(value=0.1))(y_layer)

third_layer_srl = Dense(units=unidades_ocultas_capa_3_srl,
                    use_bias=True,
                    kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=3),
                    bias_initializer=Constant(value=0.1))(y_layer)

softmax_layer_ner = Activation("softmax", name='softmax_layer')(third_layer_ner)
softmax_layer_chunking = Activation("softmax", name='softmax_layer')(third_layer_chunking)
softmax_layer_pos = Activation("softmax", name='softmax_layer')(third_layer_pos)
softmax_layer_st = Activation("softmax", name='softmax_layer')(third_layer_st)
softmax_layer_srl = Activation("softmax", name='softmax_layer')(third_layer_srl)


# Agrego las capas al modelo

model_ner = Model(inputs=[main_input, aux_input_layer], outputs=[softmax_layer_ner])
model_chunking = Model(inputs=[main_input, aux_input_layer], outputs=[softmax_layer_chunking])
model_pos = Model(inputs=[main_input, aux_input_layer], outputs=[softmax_layer_pos])
model_st = Model(inputs=[main_input, aux_input_layer], outputs=[softmax_layer_st])
model_srl = Model(inputs=[main_input, aux_input_layer], outputs=[softmax_layer_srl])


# Compilo la red

model_ner.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ner.summary()
model_chunking.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_chunking.summary()
model_pos.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_pos.summary()
model_st.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_st.summary()
model_srl.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_srl.summary()


# Entreno
inicio_carga_casos = time.time()
print 'Cargando casos de entrenamiento...'

# NER
print '--> NER...'

# Abro el archivo con casos de entrenamiento
x_train_a = []
x_train_b = []
y_train = []
with open(archivo_corpus_entrenamiento_ner, 'rb') as archivo_csv:
    lector = csv.reader(archivo_csv, delimiter=',')
    for linea in lector:
        caso = [int(x) for x in linea[1:-unidades_ocultas_capa_3_ner]]        
        y_train.append([int(x) for x in linea[-unidades_ocultas_capa_3_ner:]])
        x_train_a.append(caso)
        distancia_palabra = range(-int(linea[0]),len(caso)-(int(linea[0])))
        distancia_palabra += [np.iinfo('int32').min for _ in range(len(distancia_palabra),largo_oracion)]
        distancia_verbo = [np.iinfo('int32').min]*len(distancia_palabra)
        x_train_b.append(np.array([ np.array([x,y]) for (x,y) in zip(distancia_palabra, distancia_verbo) ]))

x_train_a_ner = pad_sequences(x_train_a, padding='post', value=indice_OUT)
x_train_b_ner = np.array(x_train_b)
y_train_ner = np.array(y_train)

# Chunking
print '--> Chunking...'

# Abro el archivo con casos de entrenamiento
x_train_a = []
x_train_b = []
y_train = []
with open(archivo_corpus_entrenamiento_chunking, 'rb') as archivo_csv:
    lector = csv.reader(archivo_csv, delimiter=',')
    for linea in lector:
        caso = [int(x) for x in linea[1:-unidades_ocultas_capa_3_chunking]]        
        y_train.append([int(x) for x in linea[-unidades_ocultas_capa_3_chunking:]])
        x_train_a.append(caso)
        distancia_palabra = range(-int(linea[0]),len(caso)-(int(linea[0])))
        distancia_palabra += [np.iinfo('int32').min for _ in range(len(distancia_palabra),largo_oracion)]
        distancia_verbo = [np.iinfo('int32').min]*len(distancia_palabra)
        x_train_b.append(np.array([ np.array([x,y]) for (x,y) in zip(distancia_palabra, distancia_verbo) ]))

x_train_a_chunking = pad_sequences(x_train_a, padding='post', value=indice_OUT)
x_train_b_chunking = np.array(x_train_b)
y_train_chunking = np.array(y_train)

# POS
print '--> POS...'

# Abro el archivo con casos de entrenamiento
x_train_a = []
x_train_b = []
y_train = []
with open(archivo_corpus_entrenamiento_pos, 'rb') as archivo_csv:
    lector = csv.reader(archivo_csv, delimiter=',')
    for linea in lector:
        caso = [int(x) for x in linea[1:-unidades_ocultas_capa_3_pos]]        
        y_train.append([int(x) for x in linea[-unidades_ocultas_capa_3_pos:]])
        x_train_a.append(caso)
        distancia_palabra = range(-int(linea[0]),len(caso)-(int(linea[0])))
        distancia_palabra += [np.iinfo('int32').min for _ in range(len(distancia_palabra),largo_oracion)]
        distancia_verbo = [np.iinfo('int32').min]*len(distancia_palabra)
        x_train_b.append(np.array([ np.array([x,y]) for (x,y) in zip(distancia_palabra, distancia_verbo) ]))

x_train_a_pos = pad_sequences(x_train_a, padding='post', value=indice_OUT)
x_train_b_pos = np.array(x_train_b)
y_train_pos = np.array(y_train)

# SuperTag
print '--> SuperTag...'

# Abro el archivo con casos de entrenamiento
x_train_a = []
x_train_b = []
y_train = []
with open(archivo_corpus_entrenamiento_st, 'rb') as archivo_csv:
    lector = csv.reader(archivo_csv, delimiter=',')
    for linea in lector:
        caso = [int(x) for x in linea[1:-unidades_ocultas_capa_3_st]]        
        y_train.append([int(x) for x in linea[-unidades_ocultas_capa_3_st:]])
        x_train_a.append(caso)
        distancia_palabra = range(-int(linea[0]),len(caso)-(int(linea[0])))
        distancia_palabra += [np.iinfo('int32').min for _ in range(len(distancia_palabra),largo_oracion)]
        distancia_verbo = [np.iinfo('int32').min]*len(distancia_palabra)
        x_train_b.append(np.array([ np.array([x,y]) for (x,y) in zip(distancia_palabra, distancia_verbo) ]))

x_train_a_st = pad_sequences(x_train_a, padding='post', value=indice_OUT)
x_train_b_st = np.array(x_train_b)
y_train_st = np.array(y_train)

# SRL
print '--> SRL...'

# Abro el archivo con casos de entrenamiento
x_train_a = []
x_train_b = []
y_train = []
with open(archivo_corpus_entrenamiento_srl, 'rb') as archivo_csv:
    lector = csv.reader(archivo_csv, delimiter=',')
    for linea in lector:
        x_train_a.append([int(x) for x in linea[2:-unidades_ocultas_capa_3_srl]])
        distancia_palabra = range(-int(linea[0]),len(linea)-(2+unidades_ocultas_capa_3_srl+int(linea[0])))
        distancia_palabra += [np.iinfo('int32').min for _ in range(len(distancia_palabra),largo_oracion)]
        distancia_verbo = range(-int(linea[1]),len(linea)-(2+unidades_ocultas_capa_3_srl+int(linea[1])))
        distancia_verbo += [np.iinfo('int32').min for _ in range(len(distancia_verbo),largo_oracion)]
        x_train_b.append(np.array([ np.array([x,y]) for (x,y) in zip(distancia_palabra, distancia_verbo) ]))
        y_train.append([int(x) for x in linea[-unidades_ocultas_capa_3_srl:]])

x_train_a_srl = pad_sequences(x_train_a, padding='post', value=indice_OUT)
x_train_b_srl = np.array(x_train_b)
y_train_srl = np.array(y_train)


print 'Cargando casos de prueba...' 

# NER
print '--> NER...'

# Abro el archivo con casos de prueba
x_test_a = []
x_test_b = []
y_test = []
with open(archivo_corpus_pruebas_ner, 'rb') as archivo_csv:
    lector = csv.reader(archivo_csv, delimiter=',')
    for linea in lector:
        caso = [int(x) for x in linea[1:-unidades_ocultas_capa_3_ner]]
        y_test.append([int(x) for x in linea[-unidades_ocultas_capa_3_ner:]])
        x_test_a.append(caso)
        distancia_palabra = range(-int(linea[0]),len(caso)-(int(linea[0])))
        distancia_palabra += [np.iinfo('int32').min for _ in range(len(distancia_palabra),largo_oracion)]
        distancia_verbo = [np.iinfo('int32').min]*len(distancia_palabra)
        x_test_b.append(np.array([ np.array([x,y]) for (x,y) in zip(distancia_palabra, distancia_verbo) ]))

x_test_a_ner = pad_sequences(x_test_a, padding='post', value=indice_OUT)
x_test_b_ner = np.array(x_test_b)
y_test_ner = np.array(y_test)

# Chunking
print '--> Chunking...'

# Abro el archivo con casos de prueba
x_test_a = []
x_test_b = []
y_test = []
with open(archivo_corpus_pruebas_chunking, 'rb') as archivo_csv:
    lector = csv.reader(archivo_csv, delimiter=',')
    for linea in lector:
        caso = [int(x) for x in linea[1:-unidades_ocultas_capa_3_chunking]]
        y_test.append([int(x) for x in linea[-unidades_ocultas_capa_3_chunking:]])
        x_test_a.append(caso)
        distancia_palabra = range(-int(linea[0]),len(caso)-(int(linea[0])))
        distancia_palabra += [np.iinfo('int32').min for _ in range(len(distancia_palabra),largo_oracion)]
        distancia_verbo = [np.iinfo('int32').min]*len(distancia_palabra)
        x_test_b.append(np.array([ np.array([x,y]) for (x,y) in zip(distancia_palabra, distancia_verbo) ]))

x_test_a_chunking = pad_sequences(x_test_a, padding='post', value=indice_OUT)
x_test_b_chunking = np.array(x_test_b)
y_test_chunking = np.array(y_test)

# POS
print '--> POS...'

# Abro el archivo con casos de prueba
x_test_a = []
x_test_b = []
y_test = []
with open(archivo_corpus_pruebas_pos, 'rb') as archivo_csv:
    lector = csv.reader(archivo_csv, delimiter=',')
    for linea in lector:
        caso = [int(x) for x in linea[1:-unidades_ocultas_capa_3_pos]]
        y_test.append([int(x) for x in linea[-unidades_ocultas_capa_3_pos:]])
        x_test_a.append(caso)
        distancia_palabra = range(-int(linea[0]),len(caso)-(int(linea[0])))
        distancia_palabra += [np.iinfo('int32').min for _ in range(len(distancia_palabra),largo_oracion)]
        distancia_verbo = [np.iinfo('int32').min]*len(distancia_palabra)
        x_test_b.append(np.array([ np.array([x,y]) for (x,y) in zip(distancia_palabra, distancia_verbo) ]))

x_test_a_pos = pad_sequences(x_test_a, padding='post', value=indice_OUT)
x_test_b_pos = np.array(x_test_b)
y_test_pos = np.array(y_test)

# SuperTag
print '--> SuperTag...'

# Abro el archivo con casos de prueba
x_test_a = []
x_test_b = []
y_test = []
with open(archivo_corpus_pruebas_st, 'rb') as archivo_csv:
    lector = csv.reader(archivo_csv, delimiter=',')
    for linea in lector:
        caso = [int(x) for x in linea[1:-unidades_ocultas_capa_3_st]]
        y_test.append([int(x) for x in linea[-unidades_ocultas_capa_3_st:]])
        x_test_a.append(caso)
        distancia_palabra = range(-int(linea[0]),len(caso)-(int(linea[0])))
        distancia_palabra += [np.iinfo('int32').min for _ in range(len(distancia_palabra),largo_oracion)]
        distancia_verbo = [np.iinfo('int32').min]*len(distancia_palabra)
        x_test_b.append(np.array([ np.array([x,y]) for (x,y) in zip(distancia_palabra, distancia_verbo) ]))

x_test_a_st = pad_sequences(x_test_a, padding='post', value=indice_OUT)
x_test_b_st = np.array(x_test_b)
y_test_st = np.array(y_test)

# SRL
print '--> SRL...'

# Abro el archivo con casos de prueba
x_test_a = []
x_test_b = []
y_test = []
with open(archivo_corpus_pruebas_srl, 'rb') as archivo_csv:
    lector = csv.reader(archivo_csv, delimiter=',')
    for linea in lector:
        x_test_a.append([int(x) for x in linea[2:-unidades_ocultas_capa_3_srl]])
        distancia_palabra = range(-int(linea[0]),len(linea)-(2+unidades_ocultas_capa_3_srl+int(linea[0])))
        distancia_palabra += [np.iinfo('int32').min for _ in range(len(distancia_palabra),largo_oracion)]
        distancia_verbo = range(-int(linea[1]),len(linea)-(2+unidades_ocultas_capa_3_srl+int(linea[1])))
        distancia_verbo += [np.iinfo('int32').min for _ in range(len(distancia_verbo),largo_oracion)]
        x_test_b.append(np.array([ np.array([x,y]) for (x,y) in zip(distancia_palabra, distancia_verbo) ]))
        y_test.append([int(x) for x in linea[-unidades_ocultas_capa_3_srl:]])

x_test_a_srl = pad_sequences(x_test_a, padding='post', value=indice_OUT)
x_test_b_srl = np.array(x_test_b)
y_test_srl = np.array(y_test)

duracion_carga_casos = time.time() - inicio_carga_casos

print 'Tiempo de carga de casos de Entrenamiento/Prueba: {0} hs, {1} min, {2} s'.format(int(duracion_carga_casos/3600),int((duracion_carga_casos % 3600)/60),int((duracion_carga_casos % 3600) % 60))


print 'Entrenando...'
inicio_entrenamiento = time.time()

history_ner = {'acc':[0]*cant_iteraciones, 'val_acc':[0]*cant_iteraciones, 'loss':[0]*cant_iteraciones, 'val_loss':[0]*cant_iteraciones}
history_chunking = {'acc':[0]*cant_iteraciones, 'val_acc':[0]*cant_iteraciones, 'loss':[0]*cant_iteraciones, 'val_loss':[0]*cant_iteraciones}
history_pos = {'acc':[0]*cant_iteraciones, 'val_acc':[0]*cant_iteraciones, 'loss':[0]*cant_iteraciones, 'val_loss':[0]*cant_iteraciones}
history_st = {'acc':[0]*cant_iteraciones, 'val_acc':[0]*cant_iteraciones, 'loss':[0]*cant_iteraciones, 'val_loss':[0]*cant_iteraciones}
history_srl = {'acc':[0]*cant_iteraciones, 'val_acc':[0]*cant_iteraciones, 'loss':[0]*cant_iteraciones, 'val_loss':[0]*cant_iteraciones}

for i in range(cant_iteraciones):
    print 'Iteracion: ' + str(i)
    print 'NER'
    history_aux = model_ner.fit({'main_input': x_train_a_ner, 'aux_input': x_train_b_ner}, 
                            {'softmax_layer': y_train_ner}, epochs=1, batch_size=100, 
                            validation_data=({'main_input': x_test_a_ner, 'aux_input': x_test_b_ner}, {'softmax_layer': y_test_ner}), 
                            verbose=0) 
    history_ner['acc'][i] = history_aux.history['acc'][0]
    history_ner['val_acc'][i] = history_aux.history['val_acc'][0]
    history_ner['loss'][i] = history_aux.history['loss'][0]
    history_ner['val_loss'][i] = history_aux.history['val_loss'][0]

    print 'CHUNKING'
    history_aux = model_chunking.fit({'main_input': x_train_a_chunking, 'aux_input': x_train_b_chunking}, 
                            {'softmax_layer': y_train_chunking}, epochs=1, batch_size=100, 
                            validation_data=({'main_input': x_test_a_chunking, 'aux_input': x_test_b_chunking}, {'softmax_layer': y_test_chunking}), 
                            verbose=0) 
    history_chunking['acc'][i] = history_aux.history['acc'][0]
    history_chunking['val_acc'][i] = history_aux.history['val_acc'][0]
    history_chunking['loss'][i] = history_aux.history['loss'][0]
    history_chunking['val_loss'][i] = history_aux.history['val_loss'][0]

    print 'POS'
    history_aux = model_pos.fit({'main_input': x_train_a_pos, 'aux_input': x_train_b_pos}, 
                            {'softmax_layer': y_train_pos}, epochs=1, batch_size=100, 
                            validation_data=({'main_input': x_test_a_pos, 'aux_input': x_test_b_pos}, {'softmax_layer': y_test_pos}), 
                            verbose=0) 
    history_pos['acc'][i] = history_aux.history['acc'][0]
    history_pos['val_acc'][i] = history_aux.history['val_acc'][0]
    history_pos['loss'][i] = history_aux.history['loss'][0]
    history_pos['val_loss'][i] = history_aux.history['val_loss'][0]

    print 'ST Reducido'
    history_aux = model_st.fit({'main_input': x_train_a_st, 'aux_input': x_train_b_st}, 
                            {'softmax_layer': y_train_st}, epochs=1, batch_size=100, 
                            validation_data=({'main_input': x_test_a_st, 'aux_input': x_test_b_st}, {'softmax_layer': y_test_st}), 
                            verbose=0) 
    history_st['acc'][i] = history_aux.history['acc'][0]
    history_st['val_acc'][i] = history_aux.history['val_acc'][0]
    history_st['loss'][i] = history_aux.history['loss'][0]
    history_st['val_loss'][i] = history_aux.history['val_loss'][0]

    print 'SRL'
    history_aux = model_srl.fit({'main_input': x_train_a_srl, 'aux_input': x_train_b_srl}, 
                            {'softmax_layer': y_train_srl}, epochs=1, batch_size=100, 
                            validation_data=({'main_input': x_test_a_srl, 'aux_input': x_test_b_srl}, {'softmax_layer': y_test_srl}), 
                            verbose=0) 
    history_srl['acc'][i] = history_aux.history['acc'][0]
    history_srl['val_acc'][i] = history_aux.history['val_acc'][0]
    history_srl['loss'][i] = history_aux.history['loss'][0]
    history_srl['val_loss'][i] = history_aux.history['val_loss'][0]


duracion_entrenamiento = time.time() - inicio_entrenamiento

print 'Obteniendo metricas...'

inicio_metricas = time.time()
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# NER
etiquetas = range(unidades_ocultas_capa_3_ner)

predictions = model_ner.predict({'main_input': x_test_a_ner, 'aux_input': x_test_b_ner}, batch_size=200, verbose=0)
y_pred = []
for p in predictions:
    p = p.tolist()
    ind_max = p.index(max(p))
    etiqueta = etiquetas[ind_max]
    y_pred.append(etiqueta)
y_true = []
for p in y_test_ner:
    p = p.tolist()
    ind_max = p.index(max(p))
    etiqueta = etiquetas[ind_max]
    y_true.append(etiqueta)
conf_mat_ner = confusion_matrix(y_true, y_pred, labels=etiquetas)
(precision_ner, recall_ner, fscore_ner, _) = precision_recall_fscore_support(y_true, y_pred)

# Chunking
etiquetas = range(unidades_ocultas_capa_3_chunking)

predictions = model_chunking.predict({'main_input': x_test_a_chunking, 'aux_input': x_test_b_chunking}, batch_size=200, verbose=0)
y_pred = []
for p in predictions:
    p = p.tolist()
    ind_max = p.index(max(p))
    etiqueta = etiquetas[ind_max]
    y_pred.append(etiqueta)
y_true = []
for p in y_test_chunking:
    p = p.tolist()
    ind_max = p.index(max(p))
    etiqueta = etiquetas[ind_max]
    y_true.append(etiqueta)
conf_mat_chunking = confusion_matrix(y_true, y_pred, labels=etiquetas)
(precision_chunking, recall_chunking, fscore_chunking, _) = precision_recall_fscore_support(y_true, y_pred)

# POS
etiquetas = range(unidades_ocultas_capa_3_pos)

predictions = model_pos.predict({'main_input': x_test_a_pos, 'aux_input': x_test_b_pos}, batch_size=200, verbose=0)
y_pred = []
for p in predictions:
    p = p.tolist()
    ind_max = p.index(max(p))
    etiqueta = etiquetas[ind_max]
    y_pred.append(etiqueta)
y_true = []
for p in y_test_pos:
    p = p.tolist()
    ind_max = p.index(max(p))
    etiqueta = etiquetas[ind_max]
    y_true.append(etiqueta)
conf_mat_pos = confusion_matrix(y_true, y_pred, labels=etiquetas)
(precision_pos, recall_pos, fscore_pos, _) = precision_recall_fscore_support(y_true, y_pred)

# SRL
etiquetas = range(unidades_ocultas_capa_3_srl)

predictions = model_srl.predict({'main_input': x_test_a_srl, 'aux_input': x_test_b_srl}, batch_size=200, verbose=0)
y_pred = []
for p in predictions:
    p = p.tolist()
    ind_max = p.index(max(p))
    etiqueta = etiquetas[ind_max]
    y_pred.append(etiqueta)
y_true = []
for p in y_test_srl:
    p = p.tolist()
    ind_max = p.index(max(p))
    etiqueta = etiquetas[ind_max]
    y_true.append(etiqueta)
conf_mat_srl = confusion_matrix(y_true, y_pred, labels=etiquetas)
(precision_srl, recall_srl, fscore_srl, _) = precision_recall_fscore_support(y_true, y_pred)

# SuperTag
etiquetas = range(unidades_ocultas_capa_3_st)

predictions = model_st.predict({'main_input': x_test_a_st, 'aux_input': x_test_b_st}, batch_size=200, verbose=0)
y_pred = []
for p in predictions:
    p = p.tolist()
    ind_max = p.index(max(p))
    etiqueta = etiquetas[ind_max]
    y_pred.append(etiqueta)
y_true = []
for p in y_test_st:
    p = p.tolist()
    ind_max = p.index(max(p))
    etiqueta = etiquetas[ind_max]
    y_true.append(etiqueta)
conf_mat_st = confusion_matrix(y_true, y_pred, labels=etiquetas)
(precision_st, recall_st, fscore_st, _) = precision_recall_fscore_support(y_true, y_pred)


duracion_metricas = time.time() - inicio_metricas


# Anoto resultados
log += '\n\nTiempo de carga de casos de Entrenamiento/Prueba: {0} hs, {1} min, {2} s'.format(int(duracion_carga_casos/3600),int((duracion_carga_casos % 3600)/60),int((duracion_carga_casos % 3600) % 60))
log += '\nDuracion del entrenamiento: {0} hs, {1} min, {2} s'.format(int(duracion_entrenamiento/3600),int((duracion_entrenamiento % 3600)/60),int((duracion_entrenamiento % 3600) % 60))
log += '\nDuracion del calculo de metricas: {0} hs, {1} min, {2} s'.format(int(duracion_metricas/3600),int((duracion_metricas % 3600)/60),int((duracion_metricas % 3600) % 60))

log += '\n\nNER\n--------'
log += '\n\nAccuracy entrenamiento inicial: ' + str(history_ner['acc'][0])
log += '\nAccuracy entrenamiento final: ' + str(history_ner['acc'][-1])
log += '\n\nAccuracy validacion inicial: ' + str(history_ner['val_acc'][0])
log += '\nAccuracy validacion final: ' + str(history_ner['val_acc'][-1])

log += '\n\nLoss entrenamiento inicial: ' + str(history_ner['loss'][0])
log += '\nLoss entrenamiento final: ' + str(history_ner['loss'][-1])
log += '\n\nLoss validacion inicial: ' + str(history_ner['val_loss'][0])
log += '\nLoss validacion final: ' + str(history_ner['val_loss'][-1])

log += '\n\nPrecision: ' + str(precision_ner)
log += '\nRecall: ' + str(recall_ner)
log += '\nMedida-F: ' + str(fscore_ner)

log += '\n\nMatriz de confusion:\n' + str(conf_mat_ner)

log += '\n\nCHUNKING\n--------'
log += '\n\nAccuracy entrenamiento inicial: ' + str(history_chunking['acc'][0])
log += '\nAccuracy entrenamiento final: ' + str(history_chunking['acc'][-1])
log += '\n\nAccuracy validacion inicial: ' + str(history_chunking['val_acc'][0])
log += '\nAccuracy validacion final: ' + str(history_chunking['val_acc'][-1])

log += '\n\nLoss entrenamiento inicial: ' + str(history_chunking['loss'][0])
log += '\nLoss entrenamiento final: ' + str(history_chunking['loss'][-1])
log += '\n\nLoss validacion inicial: ' + str(history_chunking['val_loss'][0])
log += '\nLoss validacion final: ' + str(history_chunking['val_loss'][-1])

log += '\n\nPrecision: ' + str(precision_chunking)
log += '\nRecall: ' + str(recall_chunking)
log += '\nMedida-F: ' + str(fscore_chunking)

log += '\n\nMatriz de confusion:\n' + str(conf_mat_chunking)

log += '\n\nPOS\n--------'
log += '\n\nAccuracy entrenamiento inicial: ' + str(history_pos['acc'][0])
log += '\nAccuracy entrenamiento final: ' + str(history_pos['acc'][-1])
log += '\n\nAccuracy validacion inicial: ' + str(history_pos['val_acc'][0])
log += '\nAccuracy validacion final: ' + str(history_pos['val_acc'][-1])

log += '\n\nLoss entrenamiento inicial: ' + str(history_pos['loss'][0])
log += '\nLoss entrenamiento final: ' + str(history_pos['loss'][-1])
log += '\n\nLoss validacion inicial: ' + str(history_pos['val_loss'][0])
log += '\nLoss validacion final: ' + str(history_pos['val_loss'][-1])

log += '\n\nPrecision: ' + str(precision_pos)
log += '\nRecall: ' + str(recall_pos)
log += '\nMedida-F: ' + str(fscore_pos)

log += '\n\nMatriz de confusion:\n' + str(conf_mat_pos)

log += '\n\nSuperTagging\n--------'
log += '\n\nAccuracy entrenamiento inicial: ' + str(history_st['acc'][0])
log += '\nAccuracy entrenamiento final: ' + str(history_st['acc'][-1])
log += '\n\nAccuracy validacion inicial: ' + str(history_st['val_acc'][0])
log += '\nAccuracy validacion final: ' + str(history_st['val_acc'][-1])

log += '\n\nLoss entrenamiento inicial: ' + str(history_st['loss'][0])
log += '\nLoss entrenamiento final: ' + str(history_st['loss'][-1])
log += '\n\nLoss validacion inicial: ' + str(history_st['val_loss'][0])
log += '\nLoss validacion final: ' + str(history_st['val_loss'][-1])

log += '\n\nPrecision: ' + str(precision_st)
log += '\nRecall: ' + str(recall_st)
log += '\nMedida-F: ' + str(fscore_st)

log += '\n\nMatriz de confusion:\n' + str(conf_mat_st)

log += '\n\nSRL\n--------'
log += '\n\nAccuracy entrenamiento inicial: ' + str(history_srl['acc'][0])
log += '\nAccuracy entrenamiento final: ' + str(history_srl['acc'][-1])
log += '\n\nAccuracy validacion inicial: ' + str(history_srl['val_acc'][0])
log += '\nAccuracy validacion final: ' + str(history_srl['val_acc'][-1])

log += '\n\nLoss entrenamiento inicial: ' + str(history_srl['loss'][0])
log += '\nLoss entrenamiento final: ' + str(history_srl['loss'][-1])
log += '\n\nLoss validacion inicial: ' + str(history_srl['val_loss'][0])
log += '\nLoss validacion final: ' + str(history_srl['val_loss'][-1])

log += '\n\nPrecision: ' + str(precision_srl)
log += '\nRecall: ' + str(recall_srl)
log += '\nMedida-F: ' + str(fscore_srl)

log += '\n\nMatriz de confusion:\n' + str(conf_mat_srl)

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
plt.plot(history_st['acc'])
plt.plot(history_st['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig(archivo_acc_st, bbox_inches='tight')

# summarize history for loss
plt.clf()
plt.plot(history_st['loss'])
plt.plot(history_st['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig(archivo_loss_st, bbox_inches='tight')
plt.clf()

# summarize history for accuracy
plt.plot(history_srl['acc'])
plt.plot(history_srl['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig(archivo_acc_srl, bbox_inches='tight')

# summarize history for loss
plt.clf()
plt.plot(history_srl['loss'])
plt.plot(history_srl['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig(archivo_loss_srl, bbox_inches='tight')
plt.clf()

