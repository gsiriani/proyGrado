from vector_palabras import palabras_comunes
from random import random
import tensorflow as tf
window_size = 5 # Cantidad de palabras en cada caso de prueba
vector_size = 3 # Cantidad de features para cada palabra. Coincide con la cantidad de hidden units de la primer capa
cant_palabras = 100002	# Cantidad de palabras consideradas en el diccionario
unidades_ocultas_capa_2 = 5
unidades_ocultas_capa_3 = 2

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

vectores = generar_vectores_iniciales(cant_palabras, vector_size)

sess = tf.InteractiveSession()
oracion = ["La", "casa", "del", "gato", "verde"]
matriz = obtener_matriz(oracion)

x = tf.placeholder(tf.float32, shape=[window_size, cant_palabras])
y_ = tf.placeholder(tf.float32, shape=[2])

wVectores = tf.Variable(vectores)

h_1 = tf.reshape(tf.matmul(x,wVectores),[-1,vector_size * window_size])

w_capa_2 = weight_variable([(vector_size * window_size), unidades_ocultas_capa_2])
b_capa_2 = bias_variable([unidades_ocultas_capa_2])

h_2 = tf.tanh(tf.matmul(h_1, w_capa_2) + b_capa_2)

w_capa_3 = weight_variable([unidades_ocultas_capa_2, unidades_ocultas_capa_3])
b_capa_3 = bias_variable([unidades_ocultas_capa_3])

h_3 = tf.matmul(h_2,w_capa_3) + b_capa_3

y = tf.nn.softmax(h_3)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess.run(tf.initialize_all_variables())

for i in range(10):
	print y.eval(feed_dict = {x : matriz})
	sess.run(train_step, feed_dict = {x : matriz, y_ : [0,1]})
	


