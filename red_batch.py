from vector_palabras import palabras_comunes
from random import random
import sys
import tensorflow as tf
import csv

csv.field_size_limit(sys.maxsize)

window_size = 11 # Cantidad de palabras en cada caso de prueba
vector_size = 50 # Cantidad de features para cada palabra. Coincide con la cantidad de hidden units de la primer capa
cant_palabras = 55004	# Cantidad de palabras consideradas en el diccionario
unidades_ocultas_capa_2 = 100
unidades_ocultas_capa_3 = 1
file_length = 10
batch_size = 25

# Obtenemos diccionario con las palabras a utilizar
p = palabras_comunes("es-lexicon.txt")


def generar_vectores_iniciales(cantidad, tamano):
	lista_vectores = []
	for i in range (0, cantidad):
		vector = []
		for k in range(0, tamano):
			vector.append(random())
		lista_vectores.append(vector)
	return lista_vectores

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def obtener_matriz(oracion):
	# resultado = tf.zeros([window_size, vector_size] )
	matriz = []
	for i in range (window_size):
		matriz.append([0.0] * cant_palabras)
	for i in range (window_size):
		matriz[i][p.obtener_indice(oracion[i])] = 1
	return matriz

def leer_csv(nombre):
	archivo = open(nombre, "rb")
	lector = csv.reader(archivo, delimiter=' ')
	valores = []
	filas = []
	for r in lector:
		filas.append(obtener_matriz(map(lambda x: unicode(x, encoding="utf-8"),r[:window_size])))
		valores.append(map(lambda x: int(x), r[window_size:]))
	return [filas,valores]


vectores = generar_vectores_iniciales(cant_palabras, vector_size)

sess = tf.InteractiveSession()

# Placeholders para los datos de entrada y resultados
x = tf.placeholder(tf.float32, shape=[window_size * batch_size, cant_palabras])
y_ = tf.placeholder(tf.float32, shape=[batch_size])

# Lookup table completa
wVectores = tf.Variable(vectores)

# Busqueda inicial en la lookuptable
h_1 = tf.reshape(tf.matmul(x,wVectores),[batch_size, vector_size * window_size])

# Pesos y bias de la primer capa
w_capa_2 = weight_variable([(vector_size * window_size), unidades_ocultas_capa_2])
b_capa_2 = bias_variable([unidades_ocultas_capa_2])

# Calculo de primer capa, tanh como funcion de Squashing
h_2 = tf.tanh(tf.matmul(h_1, w_capa_2) + b_capa_2)

# Pesos y bias de la segunda capa
w_capa_3 = weight_variable([unidades_ocultas_capa_2, unidades_ocultas_capa_3])
b_capa_3 = bias_variable([unidades_ocultas_capa_3])

# Calculo de segunda cpaa, sin funcion de Squashing
h_3 = tf.matmul(h_2,w_capa_3) + b_capa_3

# Softmax para calculo de salida
y = tf.nn.softmax(h_3)

# Cross entropy para cada paso del entenamiento
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

#Inicializacion de variables de TF
sess.run(tf.initialize_all_variables())

#for i in range(1):
#	print i
#	batch = []
#	for j in range(batch_size):
#		batch = batch + entrenamiento[0][j]
#	sess.run(train_step, feed_dict = {x : batch, y_ : entrenamiento[1]})


archivo = open("oraciones.csv", "r")
lineas = 0
for c in archivo:
	lineas = lineas + 1

for i in range(1):
	archivo.seek(0)
	k = 0
	while (k < lineas):
		batch = []
		batch_val = []
		for j in range(batch_size):
			r = archivo.readline().replace("\r","").split(" ")
#			while (len(r) != 12):
#				r = archivo.readline().replace("\r","").split(" ")
#				k = k + 1
#				if (k >= lineas):
#					break
#			if (k >= lineas):
#				break
			oracion = obtener_matriz(map(lambda x: unicode(x, encoding="utf-8"),r[:window_size]))
			valoracion = map(lambda x: int(x), r[window_size:])
			batch = batch + oracion
			batch_val = batch_val + valoracion
		if (k >= 7000):
			break
		k = k + batch_size
		sess.run(train_step, feed_dict = {x : batch, y_ : batch_val})
print k
#for i in range(len(entrenamiento[0])):
#	print y_test.eval(feed_dict = {x_test : entrenamiento[0][i]})
#	print entrenamiento[1][i]
#	print sess.run(cross_entropy_test, feed_dict = {x_test : entrenamiento[0][i], y_test_ : entrenamiento[1][i]})
#	print