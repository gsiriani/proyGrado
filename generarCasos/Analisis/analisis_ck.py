import sys
import os
import re
import random
from funciones_generales import correct_escape_sequences, number_filter, date_filter
from funciones_vector import generate_vector_palabra, generate_vector_cero

primer_fallo = False

funcs = {
		"suj":     0,
		"cd":      1,
		"creg":    2,
		"ci":      3,
		"cc":      4,
		"cag":     5,
		"cpred":   6,
		"atr":     7,
		"ao":      8,
		"et":      9,
		"mod":    10,
		"impers": 11,
		"pass":   12
		}
opciones = {"b" : 0, "i" : 1, "e" : 2, "s" : 3}

cant_opciones = len(opciones)
cant_tags = len(funcs)
largo_vector = cant_tags * cant_opciones
window_size = int(11)

def generate_cases(words):
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
		line += generate_vector_palabra(words[i], tags, opciones, largo_vector) + "\n"
		output.append(line)
	return output

def process_sentence(sentence_in):
	salida = ""
	for p in sentence_in:
		if salida == "":
			if p[1] == None:
				salida = "OUT"
			else:
				salida = p[1]
		else:
			if p[1] == None:
				salida += " OUT"
			else:
				salida += " " + p[1]
	salida += "\n"
	return [salida]

#	intermediate = []
#	sentence = []
#	for word in sentence_in:
#		if " " in word[0]:
#			words = word[0].split(" ")
#			first = True
#			for w in words:
#				if w != "":
#					aux_uno = number_filter(w)
#					aux_dos = date_filter(aux_uno)
#					aux_tres = correct_escape_sequences(aux_dos)
#					if first:
#						sentence.append((aux_tres, word[1], word[2]))
#						first = False
#					else:
#						sentence.append((aux_tres, word[1], False))
#		elif "_" in word[0]:
#			words = word[0].split("_")
#			first = True
#			for w in words:
#				if w != "":
#					aux_uno = number_filter(w)
#					aux_dos = date_filter(aux_uno)
#					aux_tres = correct_escape_sequences(aux_dos)
#					if first:
#						sentence.append((aux_tres,word[1],word[2]))
#						first = False
#					else:
#						sentence.append((aux_tres, word[1], False))
#		elif word[0] != "":
#			aux_uno = number_filter(word[0])
#			aux_dos = date_filter(aux_uno)
#			aux_tres = correct_escape_sequences(aux_dos)
#			sentence.append((aux_tres,word[1],word[2]))
#	length = len(sentence)
#	for i in range(length):
#		if sentence[i][1] == None:
#			intermediate.append((sentence[i][0], None))
#		elif sentence[i][2]:
#			if i < (len(sentence) - 1) and not sentence[i + 1][2]:
#				intermediate.append((sentence[i][0], sentence[i][1], "b"))
#			else:
#				intermediate.append((sentence[i][0], sentence[i][1], "s"))
#		else:
#			if i < (len(sentence) - 1) and not sentence[i + 1][2]:
#				intermediate.append((sentence[i][0], sentence[i][1], "i"))
#			else:
#				intermediate.append((sentence[i][0], sentence[i][1], "e"))
#	output = generate_cases(intermediate)
#	return output

def process_file(input_file, output_file):
	in_sentence = False
	sentence = []
	in_chunk = []
	first = True
	for line in input_file:
		if not in_sentence and "<sentence" in line:
			in_sentence = True
			first = True
			in_chunk = 0
			tag = ""
			valor = ""
		if in_sentence and in_chunk == 0:
			if (" wd=\"") in line:
				palabra = re.sub(".* wd=\"","",line)
				palabra = re.sub("\".*\n","",palabra)
				sentence.append((palabra, None, True))
			else:
				if (("<grup.verb>") in line or ("<grup.verb ") in line) and ("</grup.verb>") not in line and "/>" not in line:
					in_chunk = 1
					tag = "grup.verb"
					valor = "verbo"
				else:
					for func in funcs:
						if (" func=\"" + func + "\"") in line and "/>" not in line and "</" not in line:
							in_chunk = 1
							tag = re.sub(" .*\n","",re.sub(".*?<","",line))
							valor = func
		elif in_sentence and in_chunk > 0:
			if (" wd=\"") in line:
				palabra = re.sub(".* wd=\"","",line)
				palabra = re.sub("\".*\n","",palabra)
				palabra = palabra.decode('utf-8').lower().encode('utf-8')
				sentence.append((palabra, valor, first))
				if first:
					first = False
			elif (("<" + tag + " ") in line or ("<" + tag + ">") in line) and ("</" + tag + ">") not in line and "/>" not in line:
				in_chunk += 1
			elif ("<" + tag + " ") not in line and ("<" + tag + ">") not in line and ("</" + tag + ">") in line and "/>" not in line:
				in_chunk -= 1
				if in_chunk == 0:
					first = True
		if in_sentence and "</sentence" in line:
			in_sentence = False
			output = process_sentence(sentence)
			for o in output:
				output_file.write(o)
			sentence = []

input_folder = sys.argv[1]
output_folder = sys.argv[2]

output_training_file = open(output_folder + "/" + "chunking_training.csv","w")
output_testing_file = open(output_folder + "/" + "chunking_testing.csv","w")
output_pruebas_file = open(output_folder + "/" + "chunking_pruebas.csv","w")

input_training_file = open(input_folder + "/" + "ancora_training.xml","r")
input_testing_file = open(input_folder + "/" + "ancora_testing.xml","r")
input_pruebas_file = open(input_folder + "/" + "ancora_pruebas.xml","r")

process_file(input_training_file, output_training_file)
process_file(input_testing_file, output_testing_file)
process_file(input_pruebas_file, output_pruebas_file)

input_training_file.close()
input_pruebas_file.close()
input_testing_file.close()

output_pruebas_file.close()
output_testing_file.close()
output_training_file.close()