import sys
import re
import os

separador = ","
cantidad_previos = 1
cantidad_iobes = 4

dicc_names = {
	0 : "Persona",
	1 : "Lugar",
	2 : "Organizacion",
	3 : "Otro"
}

largo_salida = cantidad_iobes * len(dicc_names) + 1

def list_to_str(vector, separador, palabras):
	salida = ""
	primero = True
	for p in vector:
		if primero:
			salida = palabras[int(p)]
			primero = False
		else:
			salida += separador + palabras[int(p)]
	return salida

def es_out(vector):
	out = True
	for i in range(len(vector) - 1):
		if vector[i] != "0":
			out = False
	return out

def tag_to_name(tag, names, cantidad_iobes):
	for i in range(len(tag) - 1):
		if tag[i] == "1":
			return names[i / cantidad_iobes]
	return "OUT"

lista_palabras = open(sys.argv[1],"r")
a_entrada = open(sys.argv[2],"r")
a_salida = open(sys.argv[3],"w")

palabras = {}
cantidades = {}

diccionario = []
for linea in lista_palabras:
	p = linea.replace("\n","")
	diccionario.append(p)

lista_palabras.close()

for linea in a_entrada:
	lista = linea.replace("\n","").split(separador)
	largo = len(lista) - largo_salida
	oracion = list_to_str(lista[cantidad_previos:largo], " ",diccionario)
	etiquetas = lista[largo:]
	palabra = int(lista[int(lista[0]) + 1])
	if palabra not in cantidades:
		cantidades[palabra] = (0,1)
	else:
		cantidades[palabra] = (cantidades[palabra][0],cantidades[palabra][1] + 1)
	if not es_out(etiquetas):
		cantidades[palabra] = (cantidades[palabra][0] + 1,cantidades[palabra][1])
		etiqueta = tag_to_name(etiquetas, dicc_names, cantidad_iobes)
		if palabra not in palabras:
			palabras[palabra] = [oracion + " " + etiqueta]
		else:
			palabras[palabra].append(oracion + " " + etiqueta)

for palabra in palabras:
	a_salida.write(diccionario[palabra] + " " + str(cantidades[palabra][0]) + " " + str(cantidades[palabra][1]) + "\n\n----------------------------------\n\n")
	for oracion in palabras[palabra]:
		a_salida.write(oracion + "\n")
	a_salida.write("\n==================================\n\n")
a_salida.write("FIN")
a_entrada.close()
a_salida.close()