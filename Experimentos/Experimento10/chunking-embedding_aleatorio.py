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
vector_size = 50 # Cantidad de features a considerar por palabra
unidades_ocultas_capa_2 = 300
unidades_ocultas_capa_3 = 24 # SE MODIFICA PARA CADA PROBLEMA A RESOLVER

archivo_embedding = path_proyecto + "/embedding/embedding_total.txt"
archivo_corpus_entrenamiento = path_proyecto + '/corpus/Ventana/Entrenamiento/chunking_training.csv'
archivo_corpus_pruebas = path_proyecto + '/corpus/Ventana/Pruebas/chunking_pruebas.csv'

archivo_acc = path_proyecto + '/Experimentos/Experimento7/accuracy.png'
archivo_loss = path_proyecto + '/Experimentos/Experimento7/loss.png'

log = 'Log de ejecucion:\n-----------------'

# Cargo embedding inicial
palabras = palabras_comunes(archivo_embedding) # Indice de cada palabra en el diccionario
cant_palabras = len(palabras)  # Cantidad de palabras consideradas en el diccionario
print 'Cantidad de palabras consideradas: ' + str(cant_palabras)


# Defino las capas de la red

# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
embedding_layer = Embedding(input_dim=cant_palabras, output_dim=vector_size, embeddings_initializer='uniform',
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
sgd = optimizers.SGD(lr=0.3, momentum=0.05)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
model.summary()


# Entreno
inicio_carga_casos = time.time()
print 'Cargando casos de entrenamiento...'

# Abro el archivo con casos de entrenamiento
df = pd.read_csv(archivo_corpus_entrenamiento, delim_whitespace=True, skipinitialspace=True, header=None, quoting=3)

# Obtengo los indices de las palabras
largo = len(df)
for f in range(largo):
    print_progress(f, largo, prefix = 'Progreso:', suffix = 'Completado', bar_length = 50)
    for c in range(11):
        df.at[f,c]=palabras.obtener_indice(df.at[f,c])

print_progress(largo, largo, prefix = 'Progreso:', suffix = 'Completado', bar_length = 50)

# Separo features de resultados esperados
x_train = np.array(df.iloc[:largo,:11])
y_train = np.array(df.iloc[:largo,11:])


print 'Cargando casos de prueba...' 

# Abro el archivo con casos de prueba
df = pd.read_csv(archivo_corpus_pruebas, delim_whitespace=True, skipinitialspace=True, header=None, quoting=3)

# Obtengo los indices de las palabras
largo = len(df)
for f in range(largo):    
    print_progress(f, largo, prefix = 'Progreso:', suffix = 'Completado', bar_length = 50)
    for c in range(11):
        df.at[f,c]=palabras.obtener_indice(df.at[f,c])

print_progress(largo, largo, prefix = 'Progreso:', suffix = 'Completado', bar_length = 50)

# Separo features de resultados esperados
x_test = np.array(df.iloc[:largo,:11])
y_test = np.array(df.iloc[:largo,11:])

duracion_carga_casos = time.time() - inicio_carga_casos

# x_train, x_test, y_train, y_test = train_test_split(X, Y)

# x_train = np.array(x_train)
# x_test = np.array(x_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)


inicio_entrenamiento = time.time()
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=500, batch_size=100, verbose=2)
duracion_entrenamiento = time.time() - inicio_entrenamiento

# list all data in history
print(history.history.keys())

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
