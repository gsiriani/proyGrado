import sys
import os
import re
import random

ventana = 50
verificar = ["ner","supertag", "pos"]
out_tag = "OUT"
separador = ","

previo = {"ner" : "^0,",
		  "macrochunking" : "^0,",
		  "srl" : "^0,\d+,",
		  "pos" : "^0,",
		  "supertag" : "^0,",
		  "microchunking" : "^0,"}

def list_to_str(lista):
	string = ""
	for e in lista:
		if string == "":
			string = e
		else:
			string += "," + e
	return string

def transformar_a_truncada(oracion, out_ind):
	salida = ""
	lista_or = oracion.split(separador)
	cant_palabras = len(lista_or)
	for i in range(cant_palabras,ventana):
		lista_or.append(out_ind)
	for i in range(ventana):
		salida += lista_or[i] + separador
	return salida

lexicon_file = open(sys.argv[1], "r")
oraciones_file = open(sys.argv[2],"r")
repartidos_file = open(sys.argv[3],"r")
carpeta_entrada = sys.argv[4]
carpeta_salida = sys.argv[5]

if not re.match(".*/$",carpeta_entrada):
	carpeta_entrada += "/"
if not re.match(".*/$",carpeta_salida):
	carpeta_salida += "/"

out_ind = None
i = 0
for line in lexicon_file:
	if line.replace("\n","") == out_tag:
		out_ind = str(i)
		break
	i += 1
lexicon_file.close()

repartidos = ""
for linea in repartidos_file:
	repartidos += linea
repartidos_file.close()
dicc_rep = eval(repartidos)

oraciones = []
for linea in oraciones_file:
	oraciones.append(linea.replace("\n",""))
oraciones_file.close()

for nombre_archivo in os.listdir(carpeta_entrada):
	archivo_entrada = open(carpeta_entrada + nombre_archivo, "r")
	archivo_salida = open(carpeta_salida + nombre_archivo, "w")
	nombre = re.sub("_.*$","",nombre_archivo)
	if nombre in dicc_rep:
		entrenamiento = dicc_rep[nombre]
		casos = []
		if nombre in verificar:
			for linea in archivo_entrada:
				casos.append(linea)
			for caso in entrenamiento:
				i = 0
				oracion = oraciones[caso]
				transformada = transformar_a_truncada(oracion,out_ind)
				while not re.match(previo[nombre],casos[i]) or not re.match(previo[nombre] + transformada + ".*\n",casos[i]):
					i += 1
				archivo_salida.write(casos[i])
				i += 1
				while not re.match(previo[nombre],casos[i]):
					archivo_salida.write(casos[i])
					i += 1
		else:
			i = -1
			regex = previo[nombre]
			for linea in archivo_entrada:
				if re.match(regex,linea):
					i += 1
				if i in entrenamiento:
					archivo_salida.write(linea)
	archivo_entrada.close()
	archivo_salida.close()