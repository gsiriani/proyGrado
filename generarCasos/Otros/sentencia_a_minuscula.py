import sys
import os
import re
from funciones_generales import list_to_str

def lista_a_min(lista):
	nueva = []
	for par in lista:
		if par[0] == "NUM" or par[0] == "UNK" or par[0] == "DATE" or par[0] == "OUT" or par[0] == "PUNCT":
			nueva.append(par)
		else:
			nuevo_par = (par[0].decode("utf-8").lower().encode("utf-8"),par[1])
			nueva.append(nuevo_par)
	return nueva

def str_lista(lista):
	salida = "["
	for par in lista:
		if '"' in par[0]:
			if salida == "[":
				salida += "('" + par[0] + "'," + str(par[1]) + ")"
			else:
				salida += ",('" + par[0] + "'," + str(par[1]) + ")"
		else:
			if salida == "[":
				salida += "(\"" + par[0] + "\"," + str(par[1]) + ")"
			else:
				salida += ",(\"" + par[0] + "\"," + str(par[1]) + ")"
	salida += "]"
	return salida


carpeta_entrada = sys.argv[1]
carpeta_salida = sys.argv[2]

if not re.match("^.*/$",carpeta_entrada):
	carpeta_entrada += "/"
if not re.match("^.*/$",carpeta_salida):
	carpeta_salida += "/"

for n_archivo in os.listdir(carpeta_entrada):
	archivo_entrada = open(carpeta_entrada + n_archivo,"r")
	archivo_salida = open(carpeta_salida + n_archivo,"w")
	for linea in archivo_entrada:
		lista_aux = linea.replace("\n","").split(" ")
		lista = eval(lista_aux[0])
		resto = lista_aux[1:]
		nueva_lista = lista_a_min(lista)
		archivo_salida.write(str_lista(nueva_lista) + " " + list_to_str(resto) + "\n")
	archivo_entrada.close()
	archivo_salida.close()


