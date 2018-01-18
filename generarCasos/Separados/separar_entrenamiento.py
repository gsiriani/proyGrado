import sys
import os
import re
import random

repartir = ["ner", "supertag", "pos", "srl", "macrochunking", "microchunking"]
ventana = 11
medio = 6
verificar = ["ner_training.csv","supertag_reducido_training.csv", "pos_training.csv"]
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

def esta_contenida(lista_contenida, lista_contenedora):
	for e in lista_contenida:
		if e not in lista_contenedora:
			return False
	return True

lexicon_file = open(sys.argv[1], "r")
entrada = open(sys.argv[2],"r")
archivo_salida = open(sys.argv[3],"w")

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

total_entrenamiento = len(entrenamiento)
cantidad = int(total_entrenamiento / len(repartir))
tolerancia = total_entrenamiento * 10

print total_entrenamiento
print cantidad
print

salida = []
utilizadas = []

for opcion in verificar:
	archivo = open(opcion,"r")
	lineas = []
	seleccionadas = []
	fallidas = []
	for line in archivo:
		lineas.append(list_to_str(line.replace("\n","").split(separador)[:ventana]))
	for i in range(cantidad):
		j = 0
		seleccion = random.randint(0,total_entrenamiento - 1)
		ventanas = transformar_a_ventana(entrenamiento[seleccion], out_ind)
		while seleccion in fallidas or seleccion in utilizadas or not esta_contenida(ventanas, lineas):
			if seleccion not in fallidas and seleccion not in utilizadas:
				fallidas.append(seleccion)
			seleccion = random.randint(0,total_entrenamiento - 1)
			ventanas = transformar_a_ventana(entrenamiento[seleccion], out_ind)
			j += 1
			if j > tolerancia:
				print len(fallidas)
				print len(seleccionadas)
				print
				print sorted(fallidas)
				print
				print "Tolerancia excedida para " + opcion
				exit(1)
		seleccionadas.append(seleccion)
		utilizadas.append(seleccion)
	salida.append(seleccionadas)

for i in range(len(repartir) - len(verificar)):
	seleccionadas = []
	for j in range(cantidad):
		seleccion = random.randint(0,total_entrenamiento - 1)
		while seleccion in utilizadas:
			seleccion += 1
			if seleccion == total_entrenamiento:
				seleccion = 0
		seleccionadas.append(seleccion)
		utilizadas.append(seleccion)
	salida.append(seleccionadas)

print "Exito"
print

archivo_salida.write("{")
for i in range(len(salida)):
	print repartir[i]
	print len(salida[i])
	print
	if i == 0:
		archivo_salida.write("\n'" + repartir[i] + "':" + str(sorted(salida[i])))
	else:
		archivo_salida.write(",\n'" + repartir[i] + "':" + str(sorted(salida[i])))
archivo_salida.write("\n}")