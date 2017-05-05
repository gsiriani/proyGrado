path_proyecto = '/home/guille/proyGrado'

import sys
sys.path.append(path_proyecto)

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, Conv1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.initializers import TruncatedNormal, Constant
from vector_palabras import palabras_comunes
from random import uniform
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from script_auxiliares import print_progress

window_size = 11 # Cantidad de palabras en cada caso de prueba
unidades_ocultas_capa_2 = 300
unidades_ocultas_capa_3 = 16 # SE MODIFICA PARA CADA PROBLEMA A RESOLVER

archivo_embedding = path_proyecto + "/embedding/embedding_total.txt"
archivo_corpus_entrenamiento = path_proyecto + '/corpus/Oracion/Entrenamiento/ner_training.csv'
archivo_corpus_pruebas = path_proyecto + '/corpus/Oracion/Pruebas/ner_pruebas.csv'

# Cargo embedding inicial
palabras = palabras_comunes(archivo_embedding) # Indice de cada palabra en el diccionario

embedding_inicial=[]
for l in open(archivo_embedding):
    embedding_inicial.append([float(x) for x in l.split()[1:]])

vector_size = len(embedding_inicial[0]) # Cantidad de features para cada palabra. Coincide con la cantidad de hidden units de la primer capa
print 'Cantidad de features considerados: ' + str(vector_size)

# Agregamos embedding para PUNCT inicializado como el mismo embedding que ':'
indice_punct_base = palabras.obtener_indice(':')
embedding_inicial.append(list(embedding_inicial[indice_punct_base]))

# todo: agregar DATE y signos de puntuacion

# Agregamos embedding para OUT, NUM y UNK
for _ in range(3):
    features_aux = []
    for _ in range(vector_size):
        features_aux.append(uniform(-1,1))
    embedding_inicial.append(list(features_aux))

embedding_inicial = np.array(embedding_inicial)

cant_palabras = len(embedding_inicial)	# Cantidad de palabras consideradas en el diccionario
print 'Cantidad de palabras consideradas: ' + str(cant_palabras)


# Defino las capas de la red

# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
embedding_layer = Embedding(input_dim=cant_palabras, output_dim=vector_size, weights=[embedding_inicial],
                            trainable=True)

convolutive_layer = Conv1D(filters=unidades_ocultas_capa_2, kernel_size=5)

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
model.add(convolutive_layer)
model.add(GlobalMaxPooling1D())
model.add(second_layer)
model.add(Activation("tanh"))
model.add(third_layer)
# model.add(Activation("softmax"))


# Compilo la red

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
model.summary()


# Entreno
print 'Cargando casos de entrenamiento...'

# Abro el archivo con casos de entrenamiento
df = pd.read_csv(archivo_corpus_entrenamiento, delim_whitespace=True, skipinitialspace=True, header=None, quoting=3)

largo = 51

# Separo features de resultados esperados
x_train = np.array(df.iloc[:largo,:1])
y_train = np.array(df.iloc[:largo,1:])

x_train_a=[] # Matriz que almacenara indices de palabras
x_train_b=[] # Matriz que almacenara distancias a la palabra a analizar

# Obtengo los indices de las palabras
for f in range(largo):
    print_progress(f, largo, prefix = 'Progreso:', suffix = 'Completado', bar_length = 50)
    oracion = eval(x_train[f,0])
    x_train_a.append([palabras.obtener_indice(palabra) for (palabra,distancia) in oracion])
    x_train_b.append([distancia for (palabra,distancia) in oracion])
#    for c in range(len(oracion)):
#        x_train_a[f,c]=palabras.obtener_indice(x_train_a[f,c])

print_progress(largo, largo, prefix = 'Progreso:', suffix = 'Completado', bar_length = 50)
print x_train_a[48:]
x_train_a=np.array(x_train_a)
x_train_b=np.array(x_train_b)


print 'Cargando casos de prueba...' 

# Abro el archivo con casos de prueba
df = pd.read_csv(archivo_corpus_pruebas, delim_whitespace=True, skipinitialspace=True, header=None, quoting=3)

largo = 5

# Separo features de resultados esperados
x_test = np.array(df.iloc[:largo,:1])
y_test = np.array(df.iloc[:largo,1:])

x_test_a=[] # Matriz que almacenara indices de palabras
x_test_b=[] # Matriz que almacenara distancias a la palabra a analizar

# Obtengo los indices de las palabras
for f in range(largo):    
    print_progress(f, largo, prefix = 'Progreso:', suffix = 'Completado', bar_length = 50)
    oracion = eval(x_test[f,0])
    x_test_a.append([palabras.obtener_indice(palabra) for (palabra,distancia) in oracion])
    x_test_b.append([distancia for (palabra,distancia) in oracion])
 #   for c in range(len(oracion)):
  #      x_test_a[f,c]=palabras.obtener_indice(x_test_a[f,c])

print_progress(largo, largo, prefix = 'Progreso:', suffix = 'Completado', bar_length = 50)
x_test_a=np.array(x_test_a)
x_test_b=np.array(x_test_b)

# x_train, x_test, y_train, y_test = train_test_split(X, Y)

# x_train = np.array(x_train)
# x_test = np.array(x_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)



history = model.fit(x_train_a, y_train, validation_data=(x_test_a, y_test), epochs=10, batch_size=25, verbose=0)

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
