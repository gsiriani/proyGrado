import sys
import os
import re
import random

repartir = [ner, supertag, pos, srl, macrochunking, microchunking]
ventana = 11
medio = 6
verificar = ["ner_training.csv","supertag_reducido.csv"]
out_tag = "OUT"
separador = ","

def transformar_a_ventana(oracion, out_ind):
	lista_salida = []
	lista_or = oracion.split(separdor)
	cant_palabras = len(lista_or)
	for i in range(cant_palabras):
		line = ""
		max_index = min(i + medio, cant_palabras)
		min_index = max(0, i + 1 - medio)
		for j in range(0, (medio - 1) - i):
			line += out_ind + separador
		for j in range(min_index, max_index):
			if j == (ventana - 1):
				line += lista_or[i] 
			else:
				line += lista_or[i] + separador
		for j in range(medio - (cant_palabras - i)):
			if j == (ventana - 1):
				line += out_ind
			else:
				line += out_ind + separador
		lista_salida.append(line)
	return lista_salida

lexicon_file = open(sys.argv[1], "r")
entrada = open(sys.argv[2],"r")

out_ind = None
i = 0
for line in lexicon_file:
	if line.replace("\n","") == out_tag:
		out_ind = str(i)
		break
	i += 1

lexicon_file.close()

entrenamiento = []
for line in entrada:
	entrenamiento.append(line.replace("\n",""))

for opcion in verificar:
	archivo = open(opcion,"r")
	lineas = []
	for line in archivo:
		lineas.append(line.replace("\n"))