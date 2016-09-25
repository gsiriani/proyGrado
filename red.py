import sys
from vector_palabras import palabras_comunes
from random import random
import tensorflow as tf
import csv

csv.field_size_limit(sys.maxsize)


window_size = 11 # Cantidad de palabras en cada caso de prueba
vector_size = 150 # Cantidad de features para cada palabra. Coincide con la cantidad de hidden units de la primer capa
cant_palabras = 100003	# Cantidad de palabras consideradas en el diccionario
unidades_ocultas_capa_2 = 100
unidades_ocultas_capa_3 = 2
file_length = 10

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
		filas.append(obtener_matriz(map(lambda x: unicode(x, encoding="latin-1"),r[:window_size])))
		valores.append(map(lambda x: int(x), r[window_size:]))
	return [filas,valores]

def colocar_unos(oracion, matriz):
	for i in range (window_size):	
		matriz[i][p.obtener_indice(oracion[i])] = 1
	return matriz

def colocar_ceros(oracion,matriz):
	for i in range (window_size):	
		matriz[i][p.obtener_indice(oracion[i])] = 0
	return matriz

vectores = generar_vectores_iniciales(cant_palabras, vector_size)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[window_size, cant_palabras])
y_ = tf.placeholder(tf.float32, shape=[2])

wVectores = tf.Variable(vectores)

h_1 = tf.reshape(tf.matmul(x,wVectores),[-1, vector_size * window_size])

w_capa_2 = weight_variable([(vector_size * window_size), unidades_ocultas_capa_2])
b_capa_2 = bias_variable([unidades_ocultas_capa_2])

h_2 = tf.tanh(tf.matmul(h_1, w_capa_2) + b_capa_2)

w_capa_3 = weight_variable([unidades_ocultas_capa_2, unidades_ocultas_capa_3])
b_capa_3 = bias_variable([unidades_ocultas_capa_3])

h_3 = tf.matmul(h_2,w_capa_3) + b_capa_3

y = tf.nn.softmax(h_3)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

#entrenamiento = leer_csv("prueba.csv")

sess.run(tf.initialize_all_variables())

oracion = [[0]*cant_palabras for i in range(window_size)]

archivo = open("diez_porciento.csv", "rb")
lector = csv.reader(archivo, delimiter=' ')
fallidos = 0
for i in range(1):
	archivo.seek(0)
	j = 0
	for r in lector:	
		j = j + 1	
		if (len(r) != 13):
			fallidos = fallidos + 1
			j = j + 1
			continue
		print j
		j = j + 1
		oracion = colocar_unos(map(lambda x: unicode(x, encoding="utf-8"),r[:window_size]), oracion)
		valoracion = map(lambda x: int(x), r[window_size:])
		sess.run(train_step, feed_dict = {x : oracion, y_ : valoracion})
		oracion = colocar_ceros(map(lambda x: unicode(x, encoding="utf-8"),r[:window_size]), oracion)
print
print j

#archivo.seek(0)
#for r in lector:
#	oracion = obtener_matriz(map(lambda x: unicode(x, encoding="utf-8"),r[:window_size]))
#	valoracion = map(lambda x: int(x), r[window_size:])
#	print valoracion
#	print y.eval(feed_dict = {x : oracion})
#	print sess.run(cross_entropy, feed_dict = {x : oracion, y_ : valoracion})
#	print
