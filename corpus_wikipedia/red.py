from vector_palabras import palabras_comunes
from random import random
import tensorflow as tf
window_size = 5
vector_size = 3
cant_palabras = 100002
unidades_ocultas_capa_2 = 150

p = palabras_comunes("es-lexicon.txt")


def generar_vectores_iniciales(cantidad, tamano):
	lista_vectores = []
	for i in range (0, cantidad):
		vector = []
		for k in range(0, tamano):
			vector.append(random())
		lista_vectores.append(vector)
	return lista_vectores

def obtener_matriz(oracion):
	# resultado = tf.zeros([window_size, vector_size] )
	matriz = []
	for i in range (window_size):
		matriz.append([0.0] * cant_palabras)
	for i in range (window_size):
		matriz[i][p.obtener_indice(oracion[i])] = 1
	return matriz

vectores = generar_vectores_iniciales(cant_palabras, vector_size)

sess = tf.InteractiveSession()
oracion = ["La", "casa", "del", "gato", "verde"]
matriz = obtener_matriz(oracion)

x = tf.placeholder(tf.float32, shape=[window_size, cant_palabras])

wVectores = tf.Variable(vectores)

h_1 = tf.tanh(tf.reshape(tf.matmul(x,wVectores),[vector_size * window_size]))

#wConvolutional = tf.Variable(tf.zeros[])
#bConvolutional = tf.Variable(tf.zeros[])

sess.run(tf.initialize_all_variables())

print h_1.eval(feed_dict={x: matriz})