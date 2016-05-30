from vector_palabras import palabras_comunes, generar_vectores_iniciales
import tensorflow as tf
window_size = 5
vector_size = 5

p = palabras_comunes("es-lexicon.txt")
vectores = generar_vectores_iniciales(100002, vector_size)

def obtener_matriz(oracion):
	# resultado = tf.zeros([window_size, vector_size] )
	matriz = []
	for palabra in oracion:
		print palabra
		vector = vectores[p.obtener_indice(palabra)]
		matriz.append(vector)
	return matriz
sess = tf.InteractiveSession()
oracion = ["La", "casa", "del", "gato"]
matriz = obtener_matriz(oracion)
I = tf.Variable(matriz)
sess.run(tf.initialize_all_variables())
print I.eval()
