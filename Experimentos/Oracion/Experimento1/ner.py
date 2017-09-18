# -*- coding: utf-8 -*-
path_proyecto = '/home/guille/proyecto/proyGrado'

import sys
sys.path.append(path_proyecto)

from keras.models import Model
from keras.layers import Dense, Activation, Embedding, Flatten, Conv1D, Input, Concatenate
from keras.layers.pooling import GlobalMaxPooling1D
from keras.initializers import TruncatedNormal, Constant
from keras.callbacks import EarlyStopping
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
unidades_ocultas_capa_3 = 16 # SE MODIFICA PARA CADA PROBLEMA A RESOLVER

archivo_embedding = path_proyecto + "/embedding/embedding_ordenado.txt"
archivo_corpus_entrenamiento = path_proyecto + '/corpus/Oracion/Entrenamiento/ner_training.csv'
archivo_corpus_pruebas = path_proyecto + '/corpus/Oracion/Pruebas/ner_pruebas.csv'


log = 'Log de ejecucion:\n-----------------\n'
log += '\nTarea: POS'
log += '\nModelo de red: Oracion'
log += '\nEmbedding inicial: Aleatorio'
log += '\nOptimizer: adam'


# Cargo embedding inicial
palabras = palabras_comunes(archivo_embedding) # Indice de cada palabra en el diccionario
cant_palabras = len(palabras)  # Cantidad de palabras consideradas en el diccionario
print 'Cantidad de palabras consideradas: ' + str(cant_palabras)


# Defino las capas de la red

main_input = Input(shape=(None,), name='main_input')

# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
embedding_layer = Embedding(input_dim=cant_palabras, output_dim=vector_size, embeddings_initializer='uniform',
                            trainable=True)(main_input)

convolutive_layer = Conv1D(filters=unidades_ocultas_capa_2, kernel_size=5)(embedding_layer)

x_layer = GlobalMaxPooling1D()(convolutive_layer)

aux_input_layer = Input(shape=(150,), name='aux_input')

concat_layer = Concatenate()([x_layer, aux_input_layer])

second_layer = Dense(units=unidades_ocultas_capa_2,
                     use_bias=True,
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=None),
                     bias_initializer=Constant(value=0.1))(concat_layer)

y_layer = Activation("tanh")(second_layer)

third_layer = Dense(units=unidades_ocultas_capa_3,
                    use_bias=True,
                    kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=None),
                    bias_initializer=Constant(value=0.1))(y_layer)

softmax_layer = Activation("softmax")(third_layer)


# Agrego las capas al modelo

model = Model(inputs=[main_input, aux_input_layer], outputs=[softmax_layer])



# model.add(embedding_layer)
# model.add(convolutive_layer)
# model.add(Concatenate())
# model.add(GlobalMaxPooling1D())
# model.add(second_layer)
# model.add(Activation("tanh"))
# model.add(third_layer)
# # model.add(Activation("softmax"))


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
# df = pd.read_csv(archivo_corpus_entrenamiento, sep=',', skipinitialspace=True, header=None, quoting=3)
# largo = len(df)

# # Separo features de resultados esperados
# x_train = np.array(df.iloc[:largo,:-unidades_ocultas_capa_3])
# y_train = np.array(df.iloc[:largo,-unidades_ocultas_capa_3:])

x_train_a = [l[1:] for l in x_train]
x_train_b = [ [i-l[0] for i in range(len(l))] for l in x_train] # Matriz que almacenara distancias a la palabra a analizar


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
x_test_b = [ [i-l[0] for i in range(len(l))] for l in x_test] # Matriz que almacenara distancias a la palabra a analizar

duracion_carga_casos = time.time() - inicio_carga_casos


print 'Entrenando...'
inicio_entrenamiento = time.time()

early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=0, mode='auto')
history = model.fit(main_input=x_train_a, aux_input_layer=x_train_b, y=y_train, epochs=10, batch_size=25, callbacks=[early_stop], verbose=2)
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

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
