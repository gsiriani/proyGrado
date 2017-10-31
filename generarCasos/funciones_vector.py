# Funciones generales de chunking
from funciones_generales import list_to_str, sumatoria

def obtener_indice(tag, tags, opcion, opciones):
	indice = tags[tag] * len(opciones) + opciones[opcion]
	return indice

def obtener_indices(valores, tags):
	indices = []
	largos = []
	for tag in tags:
		largos.append(len(tag))
	for i in range(len(valores)):
		indice = sumatoria(largos[:i]) + tags[i][valores[i]]
		indices.append(indice)
	return indices

def vector_variante(indice, largo_vector):
	vector = []
	for i in range(largo_vector):
		if i == indice:
			vector.append(1)
		else:
			vector.append(0)
	return vector

def vector_variante_multiple(indices, largo_vector):
	vector = []
	for i in range(largo_vector):
		if i in indices:
			vector.append(1)
		else:
			vector.append(0)
	return vector

def generate_vector_palabra(palabra, tags, opciones, largo_vector):
	if palabra[1] != None:
		indice = obtener_indice(palabra[1], tags, palabra[2], opciones)
		vector = vector_variante(indice, largo_vector)
		return list_to_str(vector)
	else:
		vector = vector_variante(-1, largo_vector)
		return list_to_str(vector)

def generate_vector_palabra_multiple(valores, tags, largo_vector):
		indices = obtener_indices(valores, tags)
		vector = vector_variante_multiple(indices, largo_vector)
		return list_to_str(vector)

def generate_vector_cero(largo):
	l = []
	for i in range(largo):
		l.append(0)
	return l

def put_one(output, num_tag, cantidad_opciones, opciones_tags, orden_tags, valor):
	#print orden_tags[num_tag]
	posicion = sumatoria(cantidad_opciones[:num_tag]) + opciones_tags[orden_tags[num_tag]][valor]
	output[posicion] = 1
	return output

def generate_srl(word, cantidad_opciones, opciones_tags, orden_tags):
	srl = generate_vector_cero(sumatoria(cantidad_opciones))
	for i in range(len(orden_tags)):
		if word[i + 1] != None:
			srl = put_one(srl, i, cantidad_opciones, opciones_tags, orden_tags, word[i + 1])
	output = list_to_str(srl)
	return output