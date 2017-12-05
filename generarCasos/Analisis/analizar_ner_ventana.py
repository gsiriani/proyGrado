import sys
import re
import os

separador = ","
largo_ventana = 11
clave = (largo_ventana - 1) / 2

def list_to_str(vector):
	salida = ""
	primero = True
	for p in vector:
		if primero:
			salida = str(p)
			primero = False
		else:
			salida += separador + str(p)
	return salida

def es_out(vector):
	out = True
	for i in range(len(vector) - 1):
		if vector[i] != "0":
			out = False
	return out

lista_palabras = open(sys.argv[1],"r")
a_entrada = open(sys.argv[2],"r")
a_salida = open(sys.argv[3],"w")

tags = {}
cantidad_palabras = {}
casos_ner = 0

diccionario = []
for linea in lista_palabras:
	p = linea.replace("\n","")
	diccionario.append(p)

lista_palabras.close()

for linea in a_entrada:
	lista = linea.replace("\n","").split(separador)
	etiquetas = lista[largo_ventana:]
	if not es_out(etiquetas):
		casos_ner += 1
		etiqueta = list_to_str(etiquetas)
		indice = int(lista[clave])
		if indice not in cantidad_palabras:
			cantidad_palabras[indice] = 1
		else:
			cantidad_palabras[indice] += 1
		if etiqueta not in tags:
			tags[etiqueta] = {}
		if indice not in tags[etiqueta]:
			tags[etiqueta][indice] = 1
		else:
			tags[etiqueta][indice] += 1
for tag in tags:
	a_salida.write(tag + "\n\n----------------------------------\n\n")
	for palabra in tags[tag]:
		a_salida.write(diccionario[palabra] + " " + str(tags[tag][palabra]) + " " + str(cantidad_palabras[palabra]) + "\n")
	a_salida.write("\n==================================\n\n")
a_salida.write(str(casos_ner))
a_entrada.close()
a_salida.close()