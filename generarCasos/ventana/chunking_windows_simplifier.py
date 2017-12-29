import sys
import re
import os

cambios_indice = {4 : 2, 5 : 3 ,6 : 4, 7 : 5, 8 : 6, 9 : 7, 10 : 8, 11 : 9, 12 : 10, 13 : 11}
cant_eliminar = 2
separador = ","

def cambiar_uno(indice_uno):
	if indice_uno in cambios_indice:
		return cambios_indice[indice_uno]
	else:
		return indice_uno

def generate_vector(indice_uno, largo):
	salida = []
	for i in range(largo):
		if i == indice_uno:
			salida.append("1")
		else:
			salida.append("0")
	return salida

def encontrar_uno(etiqueta):
	indice = None
	for i in range(len(etiqueta)):
		if etiqueta[i] == "1":
			indice = i
	return indice

def cambio_etiquetas(etiqueta, cant_etiqueas):
	indice_uno = encontrar_uno(etiqueta)
	indice_uno = cambiar_uno(indice_uno)
	nuevo_largo = cant_etiqueas - cant_eliminar
	vector = generate_vector(indice_uno, nuevo_largo)
	return vector

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

c_entrada = sys.argv[1]
c_salida = sys.argv[2]

if not re.match(".*/$",c_entrada):
	c_entrada += "/"
if not re.match(".*/$",c_salida):
	c_salida += "/"

for n_archivo in os.listdir(c_entrada):
	a_entrada = open(c_entrada + n_archivo,"r")
	a_salida = open(c_salida + n_archivo,"w")
	for linea in a_entrada:
		lista = linea.replace("\n","").split(separador)
		oracion = lista[:11]
		etiquetas = lista[11:]
		cant_etiquetas = len(etiquetas)
		nuevas_etiquetas = cambio_etiquetas(etiquetas, cant_etiquetas)
		oracion += nuevas_etiquetas
		a_salida.write(list_to_str(oracion) + "\n")
	a_entrada.close()
	a_salida.close()