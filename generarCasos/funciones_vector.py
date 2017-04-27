# Funciones generales de chunking
from funciones_generales import list_to_str

def obtener_indice(tag, tags, opcion, opciones):
	indice = tags[tag] * len(opciones) + opciones[opcion]
	return indice

def vector_variante(indice, largo_vector):
	vector = []
	for i in range(largo_vector):
		if i == indice:
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

def generate_vector_cero(largo):
	l = []
	for i in range(largo):
		l.append(0)
	return l