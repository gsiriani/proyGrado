import sys
import re
from funciones_generales import number_filter, date_filter

window_size = int(11)

def generate_vector_salida(supertag, supertags):
	salida = ""
	for i in range(len(supertags)):
		if i == supertags[supertag]:
			if salida == "":
				salida = "1"
			else:
				salida += " 1"
		else:
			if salida == "":
				salida = "0"
			else:
				salida += " 0"
	return salida

def generate_cases(words,dicc_st):
	output = []
	largo = len(words)
	for i in range(largo):
		line = "["
		for j in range(largo):
			if j > 0
				line += ","
			if "\"" in words[j][0]:
				line += "('" + words[j][0] + "'," + str(i - j) + ")"
			else:
				line += "(\"" + words[j][0] + "\"," + str(i -j) + ")"
		for j in range(largo,5):
			line += ",(\"OUT\"," + str(i - j) + ")"
		line += "] " + generate_vector_salida(words[i][1],dicc_st) + "\n"
		output.append(line)
	return output

archivo = open(sys.argv[1],"r")
supertags = {}
for linea in archivo:
	if linea != "\n":
		supertag = linea.replace("\n","")
		supertags[supertag] = len(supertags)
archivo.close()

archivo = open(sys.argv[2],"r")
salida = open(sys.argv[3],"w")

oracion = []
for linea in archivo:
	linea = linea.replace("\n","").replace("\r","")
	if linea == "":
		output = generate_cases(oracion, supertags)
		for o in output:
			salida.write(o)
		oracion = []
	else:
		separada = linea.split("\t")
		palabra = separada[1]
		supertag = separada[4]
		if "_" in palabra:
			palabras = palabra.split("_")
			for p in palabras:
				p = number_filter(p)
				p =	date_filter(p)
				oracion.append((p,supertag))
		elif " " in palabra:
			palabras = palabra.split(" ")
			for p in palabras:
				p = number_filter(p)
				p =	date_filter(p)
				oracion.append((p,supertag))
		else:
			palabra = number_filter(palabra)
			palabra = date_filter(palabra)
			oracion.append((palabra,supertag))

archivo.close()
salida.close()