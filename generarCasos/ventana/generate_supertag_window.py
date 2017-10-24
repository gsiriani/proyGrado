import sys
import re

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
	mitad_ventana = int(window_size / 2)
	mal = False
	for i in range(len(words)):
		line = ""
		max_index = min(i + mitad_ventana + 1,len(words))
		min_index = max(0,i - mitad_ventana)
		for j in range(0,mitad_ventana - i):
			line += "OUT "
		for j in range(min_index, max_index):
			line += words[j][0] + " "
		for j in range(6 - (len(words) - i)):
			line += "OUT "
		if len(line.split(" ")) != 12:
			mal = True
		line += generate_vector_salida(words[i][1],dicc_st) + "\n"
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
	if linea == "\n":
		output = generate_cases(oracion, supertags)
		for o in output:
			salida.write(o)
		oracion = []
	else:
		separada = linea.replace("\n","").split("\t")
		palabra = separada[1]
		supertag = separada[4]
		if "_" in palabra:
			palabras = palabra.split("_")
			for p in palabras:
				oracion.append((p,supertag))
		elif " " in palabra:
			palabras = palabra.split(" ")
			for p in palabras:
				oracion.append((p,supertag))
		else:
			oracion.append((palabra,supertag))

archivo.close()
salida.close()