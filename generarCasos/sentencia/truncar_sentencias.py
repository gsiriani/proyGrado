import sys
import re
import os

separador = ","
cantidad = 50
largos_salida = {"ner" : 17,
				"chunking" : 25,
				"chunking_reducido" : 9,
				"chunking_simplificado" : 13,
				"srl_simple" : 33,
				"pos_simple" : 12,
				"supertag_completo" : 947,
				"supertag_reducido" : 643}
largo_previo = {"ner" : 1,
				"chunking" : 1,
				"chunking_reducido" : 1,
				"chunking_simplificado" : 1,
				"srl_simple" : 2,
				"pos_simple" : 1,
				"supertag_completo" : 1,
				"supertag_reducido" : 1}

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

def obtener_nombre(n_archivo):
	return re.sub("_training.*$","",re.sub("_testing.*$","",re.sub("_pruebas.*$","",n_archivo)))

c_entrada = sys.argv[1]
c_salida = sys.argv[2]

if not re.match(".*/$",c_entrada):
	c_entrada += "/"
if not re.match(".*/$",c_salida):
	c_salida += "/"

for c_interna in os.listdir(c_entrada):
	if not re.match(".*/$",c_interna):
		c_interna += "/"
	directorio = c_entrada + c_interna
	directorio_salida = c_salida + c_interna
	for n_archivo in os.listdir(directorio):
		a_entrada = open(directorio + n_archivo,"r")
		a_salida = open(directorio_salida + n_archivo,"w")
		nombre = obtener_nombre(n_archivo)
		for linea in a_entrada:
			lista = linea.replace("\n","").split(separador)
			largo_salida = largos_salida[nombre]
			largo = len(lista) - largo_salida
			oracion = lista[:largo]
			etiquetas = lista[largo:]
			salida = []
			if int(oracion[0]) < cantidad:
				for i in range(min([cantidad + largo_previo[nombre],largo])):
					salida.append(oracion[i])
				salida += etiquetas
				a_salida.write(list_to_str(salida) + "\n")
		a_entrada.close()
		a_salida.close()