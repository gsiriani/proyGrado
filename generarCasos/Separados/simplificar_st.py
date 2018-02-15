import sys
import os
import re

largo_oracion = 11
separador = ","

def encontrar_uno(etiqueta):
	indice = None
	for i in range(len(etiqueta)):
		if etiqueta[i] == "1":
			indice = i
	return indice

def encontrar_indice(etiqueta, lista):
	indice = None
	buscar = ""
	if etiqueta in lista:
		buscar = etiqueta
	elif re.match("^v.+$",etiqueta):
		buscar = "v-unknown"
	elif re.match("^n.+$",etiqueta):
		buscar = "n-unknown"
	elif re.match("^a.+$",etiqueta):
		buscar = "a-unknown"
	for i in range(len(lista)):
		if lista[i] == buscar:
			indice = i
			break
	return indice

def crear_vector(indice, largo):
	vector = []
	for i in range(largo):
		if i == indice:
			vector.append("1")
		else:
			vector.append("0")
	return vector

def list_to_str(lista, separador):
	salida = ""
	for e in lista:
		if salida == "":
			salida = e
		else:
			salida += separador + e
	return salida + "\n"

f_viejo_dicc = open(sys.argv[1],"r")
f_nuevo_dicc = open(sys.argv[2],"r")
carpeta_entrada = sys.argv[3]
carpeta_salida = sys.argv[4]

if not re.match("./$",carpeta_entrada):
	carpeta_entrada += "/"
if not re.match("./$",carpeta_salida):
	carpeta_salida += "/"

viejo_dicc = []
for t in f_viejo_dicc:
	viejo_dicc.append(t.replace("\n",""))
nuevo_dicc = []
for t in f_nuevo_dicc:
	nuevo_dicc.append(t.replace("\n",""))

etiquetas = {}

for nombre in os.listdir(carpeta_entrada):
	print nombre
	archivo_entrada = open(carpeta_entrada + nombre, "r")
	archivo_salida = open(carpeta_salida + nombre, "w")
	for linea in archivo_entrada:
		try:
			lista = linea.replace("\n","").split(separador)
			oracion = lista[:largo_oracion]
			vector_etiqueta = lista[largo_oracion:]
			indice = encontrar_uno(vector_etiqueta)
			vieja_etiqueta = viejo_dicc[indice]
			nuevo_indice = encontrar_indice(vieja_etiqueta, nuevo_dicc)
			nuevo_vector = crear_vector(nuevo_indice,len(nuevo_dicc))
			lista_salida = oracion + nuevo_vector
			salida = list_to_str(lista_salida, separador)
			archivo_salida.write(salida)
		except:
			print linea
	archivo_entrada.close()
	archivo_salida.close()
