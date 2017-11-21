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
from random import uniform
import csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from script_auxiliares import print_progress
import time
from codecs import open, BOM_UTF8
from vector_palabras import palabras_comunes

vector_size = 150 # Cantidad de features a considerar por palabra
unidades_ocultas_capa_2 = 300
unidades_ocultas_capa_3 = 947 # SE MODIFICA PARA CADA PROBLEMA A RESOLVER

archivo_embedding = path_proyecto + "/embedding/embedding_ordenado.txt"
archivo_lexicon = path_proyecto + "/embedding/lexicon_total.txt"
archivo_corpus_entrenamiento = path_proyecto + '/corpus/Sentencia_truncada/Entrenamiento/supertag_completo_training.csv'
archivo_corpus_pruebas = path_proyecto + '/corpus/Sentencia_truncada/Pruebas/supertag_completo_pruebas.csv'

archivo_acc = './accuracy.png'
archivo_loss = './loss.png'

cant_iteraciones = 200

log = 'Log de ejecucion:\n-----------------\n'
log += '\nTarea: SuperTaggging Completo'
log += '\nModelo de red: Oracion'
log += '\nEmbedding inicial: Precalculado'
log += '\nOptimizer: adam'


# Cargo embedding inicial
palabras = palabras_comunes(archivo_lexicon) # Indice de cada palabra en el diccionario
indice_OUT = palabras.obtener_indice("OUT")
embedding_inicial = []
for l in open(archivo_embedding):
    embedding_inicial.append(list([float(x) for x in l.split()])) 

embedding_inicial = np.array(embedding_inicial)

cant_palabras = len(embedding_inicial) # Cantidad de palabras consideradas en el diccionario
print 'Cantidad de palabras consideradas: ' + str(cant_palabras)


# Defino las capas de la red

main_input = Input(shape=(None,), name='main_input')

aux_input_layer = Input(shape=(None,1), name='aux_input')

# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
embedding_layer = Embedding(input_dim=cant_palabras, output_dim=vector_size, weights=[embedding_inicial],
                            trainable=True)(main_input)

concat_layer = Concatenate()([embedding_layer, aux_input_layer])

convolutive_layer = Conv1D(filters=unidades_ocultas_capa_2, kernel_size=5)(concat_layer)

x_layer = GlobalMaxPooling1D()(convolutive_layer)

second_layer = Dense(units=unidades_ocultas_capa_2,
                     use_bias=True,
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=2),
                     bias_initializer=Constant(value=0.1))(x_layer)

y_layer = Activation("tanh")(second_layer)

third_layer = Dense(units=unidades_ocultas_capa_3,
                    use_bias=True,
                    kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=3),
                    bias_initializer=Constant(value=0.1))(y_layer)

softmax_layer = Activation("softmax", name='softmax_layer')(third_layer)


# Agrego las capas al modelo

model = Model(inputs=[main_input, aux_input_layer], outputs=[softmax_layer])
#model = Model(inputs=[main_input], outputs=[softmax_layer])


# Compilo la red

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# Entreno
inicio_carga_casos = time.time()
print 'Cargando casos de entrenamiento...'

# Abro el archivo con casos de entrenamiento
x_train = []
y_train = []
with open(archivo_corpus_entrenamiento, 'rb') as archivo_csv:
    lector = csv.reader(archivo_csv, delimiter=',')
    for linea in lector:
        x_train.append([int(x) for x in linea[:-unidades_ocultas_capa_3]])
        y_train.append([int(x) for x in linea[-unidades_ocultas_capa_3:]])

#x_train_a = [np.array(l[1:]) for l in x_train]
#x_train_b = [ np.matrix([[i-l[0]] for i in range(len(l)-1)]) for l in x_train] # Matriz que almacenara distancias a la palabra a analizar
x_train_a = [l[1:] for l in x_train]
x_train_b = [ [[i-l[0]] for i in range(len(l)-1)] for l in x_train] # Matriz que almacenara distancias a la palabra a analizar
x_train_a = pad_sequences(x_train_a, padding='post', value=indice_OUT)
x_train_b = pad_sequences(x_train_b, padding='post', value=np.iinfo('int32').min)
y_train = np.array(y_train)


print 'Cargando casos de prueba...' 

# Abro el archivo con casos de prueba
x_test = []
y_test = []
with open(archivo_corpus_pruebas, 'rb') as archivo_csv:
    lector = csv.reader(archivo_csv, delimiter=',')
    for linea in lector:
        x_test.append([int(x) for x in linea[:-unidades_ocultas_capa_3]])
        y_test.append([int(x) for x in linea[-unidades_ocultas_capa_3:]])

x_test_a = [l[1:] for l in x_test]
x_test_b = [ [[i-l[0]] for i in range(len(l)-1)] for l in x_test] # Matriz que almacenara distancias a la palabra a analizar
x_test_a = pad_sequences(x_test_a, padding='post', value=indice_OUT)
x_test_b = pad_sequences(x_test_b, padding='post', value=np.iinfo('int32').min)
y_test = np.array(y_test)

duracion_carga_casos = time.time() - inicio_carga_casos


print 'Entrenando...'
inicio_entrenamiento = time.time()

early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=0, mode='auto')
history = model.fit({'main_input': x_train_a, 'aux_input': x_train_b}, {'softmax_layer': y_train}, epochs=cant_iteraciones, batch_size=100, 
    validation_data=({'main_input': x_test_a, 'aux_input': x_test_b}, {'softmax_layer': y_test}), verbose=2)
#history = model.fit({'main_input': x_train_a}, {'softmax_layer': y_train}, epochs=10, batch_size=25, verbose=2)
duracion_entrenamiento = time.time() - inicio_entrenamiento


print 'Obteniendo metricas...'

inicio_metricas = time.time()
etiquetas = range(unidades_ocultas_capa_3)
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

predictions = model.predict({'main_input': x_test_a, 'aux_input': x_test_b}, batch_size=200, verbose=0)
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


# Anoto resultados
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