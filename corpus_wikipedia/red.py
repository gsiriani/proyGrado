from vector_palabras import palabras_comunes, generar_vectores_iniciales
import tensorflow as tf
window_size = 5
vector_size = 3
cant_palabras = 100002

p = palabras_comunes("es-lexicon.txt")
vectores = generar_vectores_iniciales(cant_palabras, vector_size)

def obtener_matriz(oracion):
	# resultado = tf.zeros([window_size, vector_size] )
	matriz = []
	for i in range (window_size):
		matriz.append([0] * cant_palabras)
	for i in range (window_size):
		matriz[i][p.obtener_indice(oracion[i])] = 1
	return tf.Variable(matriz)

sess = tf.InteractiveSession()
oracion = ["La", "casa", "del", "gato", "verde"]
matriz = obtener_matriz(oracion)

#x = tf.placeholder(tf.float32, shape=[window_size, cant_palabras])
#
#
#wVectores = tf.Variable(vectores)
#
#y = tf.nn.softmax(tf.matmul(x,wVectores))
#
#sess.run(tf.initialize_all_variables())
#print y.eval(feed_dict={x: matriz})



