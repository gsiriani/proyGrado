import sys
import os
import re
import random

ventana = 11
medio = 6
verificar = ["ner","supertag", "pos"]
out_tag = "OUT"
separador = ","

def list_to_str(lista):
	string = ""
	for e in lista:
		if string == "":
			string = e
		else:
			string += "," + e
	return string

def transformar_a_ventana(oracion, out_ind):
	lista_salida = []
	lista_or = oracion.split(separador)
	cant_palabras = len(lista_or)
	for i in range(cant_palabras):
		line = ""
		max_index = min(i + medio, cant_palabras)
		min_index = max(0, i + 1 - medio)
		for j in range(0, (medio - 1) - i):
			if line == "":
				line += out_ind
			else:
				line += separador + out_ind
		for j in range(min_index, max_index):
			if line == "":
				line += lista_or[j] 
			else:
				line += separador + lista_or[j]
		for j in range(medio - (cant_palabras - i)):
			if line == "":
				line += out_ind
			else:
				line += separador + out_ind
		lista_salida.append(line)
	return lista_salida

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
		for linea in archivo_entrada:
			casos.append(linea)
		for caso in entrenamiento:
			oracion = oraciones[caso]
			transformada = transformar_a_ventana(oracion,out_ind)
			for t in transformada:
				i = 0
				print t
				while not re.match("^" + t + ",.*\n",casos[i]):
					print i
					i += 1
				archivo_salida.write(casos[i])
	archivo_entrada.close()
	archivo_salida.close()
