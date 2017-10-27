import sys
import re
import os

carpeta_separadas = sys.argv[1]
ifolder = sys.argv[2]
ofolder = sys.argv[3]
ventana = 11
separador = " "

def extraer_oracion(linea):
	lista_oracion = linea.replace("\n","").split(separador)[:ventana]
	oracion = lista_oracion[0]
	for i in range(1,ventana):
		oracion += separador + lista_oracion[i]
	return oracion

def extraer_oraciones(archivo):
	retorno = []
	for line in archivo:
		retorno.append(extraer_oracion(line))
	return retorno

if not re.match(".*/$",carpeta_separadas):
	carpeta_separadas += "/"
if not re.match(".*/$",ifolder):
	ifolder += "/"
if not re.match(".*/$",ofolder):
	ofolder += "/"

entrenamiento = []
pruebas = []
testing = []

for sfile in os.listdir(carpeta_separadas):
	archivo = open(carpeta_separadas + sfile,"r")
	if re.match(".*_training.csv",sfile):
		entrenamiento = extraer_oraciones(archivo)
	elif re.match(".*_pruebas.csv",sfile):
		pruebas = extraer_oraciones(archivo)
	elif re.match(".*_testing.csv",sfile):
		testing = extraer_oraciones(archivo)
	archivo.close()

for archivo in os.listdir(ifolder):
	archivo_entrada = open(ifolder + archivo,"r")
	archivo_entrenamiento = open(ofolder + archivo.replace(".csv","") + "_training.csv","w")
	archivo_pruebas = open(ofolder + archivo.replace(".csv","") + "_pruebas.csv","w")
	archivo_testing = open(ofolder + archivo.replace(".csv","") + "_testing.csv","w")
	archivo_error = open(ofolder + archivo.replace(".csv","") + "_error.csv","w")
	for l in archivo_entrada:
		oracion = extraer_oracion(l)
		if oracion in entrenamiento:
			archivo_entrenamiento.write(l)
		elif oracion in pruebas:
			archivo_pruebas.write(l)
		elif oracion in testing:
			archivo_testing.write(l)
		else:
			archivo_error.write(l)
	archivo_entrada.close()
	archivo_entrenamiento.close()
	archivo_testing.close()
	archivo_pruebas.close()