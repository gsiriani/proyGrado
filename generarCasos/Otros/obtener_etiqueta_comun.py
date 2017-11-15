import sys
import re
import os

separador = ","
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

dicc_salidas = {}

for c_interna in os.listdir(c_entrada):
	if not re.match(".*/$",c_interna):
		c_interna += "/"
	directorio = c_entrada + c_interna
	for n_archivo in os.listdir(directorio):
		a_entrada = open(directorio + n_archivo,"r")
		nombre = obtener_nombre(n_archivo)
		if nombre not in dicc_salidas:
			dicc_salidas[nombre] = {}
		for linea in a_entrada:
			lista = linea.replace("\n","").split(separador)
			largo_salida = largos_salida[nombre]
			largo = len(lista) - largo_salida
			etiqueta = list_to_str(lista[largo:])
			indice_palabra = int(lista[0]) + largo_previo[nombre]
			palabra = int(lista[indice_palabra])
			if palabra not in dicc_salidas[nombre]:
				dicc_salidas[nombre][palabra] = {}
			if etiqueta not in dicc_salidas[nombre][palabra]:
				dicc_salidas[nombre][palabra][etiqueta] = 0
			dicc_salidas[nombre][palabra][etiqueta] += 1
		a_entrada.close()
for fuente in dicc_salidas:
	a_salida = open(c_salida + fuente + ".txt","w")
	for palabra in dicc_salidas[fuente]:
		max_etiqueta = ''
		max_value = 0
		for etiqueta in dicc_salidas[fuente][palabra]:
			if dicc_salidas[fuente][palabra][etiqueta] > max_value:
				max_value = dicc_salidas[fuente][palabra][etiqueta]
				max_etiqueta = etiqueta
			elif dicc_salidas[fuente][palabra][etiqueta] == max_value:
				max_etiqueta += ":" + etiqueta
		a_salida.write(str(palabra) + ":" + max_etiqueta + ":" + str(max_value) + "\n")
