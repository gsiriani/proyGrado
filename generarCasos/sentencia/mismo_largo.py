import sys
import re
import os

separador = ","
cantidad = 50
out_tag = "56945"
largos_salida = {"ner" : 9,
				"macrochunking" : 11,
				"srl" : 17,
				"pos" : 12,
				"supertag_completo" : 947,
				"supertag_reducido" : 643,
				"microchunking" : 11}
largo_previo = {"ner" : 1,
				"macrochunking" : 1,
				"srl" : 2,
				"pos" : 1,
				"supertag_completo" : 1,
				"supertag_reducido" : 1,
				"microchunking" : 1}

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

def generate_out(cant):
	salida = []
	for i in range(cant):
		salida.append(out_tag)
	return salida

def obtener_nombre(n_archivo):
	return re.sub("_training.*$","",re.sub("_testing.*$","",re.sub("_pruebas.*$","",n_archivo)))

c_entrada = sys.argv[1]
c_salida = sys.argv[2]

if not re.match(".*/$",c_entrada):
	c_entrada += "/"
if not re.match(".*/$",c_salida):
	c_salida += "/"


for n_archivo in os.listdir(c_entrada):
	a_entrada = open(c_entrada + n_archivo,"r")
	a_salida = open(c_salida + n_archivo,"w")
	nombre = obtener_nombre(n_archivo)
	for linea in a_entrada:
		lista = linea.replace("\n","").split(separador)
		largo_salida = largos_salida[nombre]
		largo = len(lista) - largo_salida
		oracion = lista[:largo]
		etiquetas = lista[largo:]
		diferencia = cantidad - (len(oracion) - largo_previo[nombre])
		oracion += generate_out(diferencia)
		oracion += etiquetas
		a_salida.write(list_to_str(oracion) + "\n")
	a_entrada.close()
	a_salida.close()