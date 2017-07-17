# Funciones generales
import re

def sumatoria(arreglo):
	total = 0
	for i in arreglo:
		total += i
	return total

def list_to_str(vector):
	salida = ""
	primero = True
	for p in vector:
		if primero:
			salida = str(p)
			primero = False
		else:
			salida += " " + str(p)
	return salida

def list_to_str_utf8(vector):
	salida = ""
	primero = True
	for p in vector:
		if primero:
			salida = p
			primero = False
		else:
			salida += " " + p
	return salida

def correct_escape_sequences(word):
	return word.replace("&quot;","\"").replace("&lt;","<").replace("&gt;",">").replace("&amp;","&")

def number_filter(word):
	try:
		num = float(word)
		return "NUM"
	except:
		return word

def date_filter(word):
	if re.match("\d+/\d+/\d+",word) or re.match("\d+-\d+-\d+",word):
		return "DATE"
	else:
		return word