import sys
import os
import re
import random
from funciones_generales import correct_escape_sequences, number_filter, date_filter
from funciones_vector import generate_vector_cero, generate_vector_palabra

tags = {"sn" : 0, "sa" : 1, "sp" : 2, "sadv" : 3, "grup.verb": 4}
opciones = {"b" : 0, "i" : 1}
cambios = {"s.a" : "sa"}

in_tag = {}
for tag in tags:
	in_tag[tag] = 0

cant_opciones = len(opciones)
cant_tags = len(tags)
largo_vector = cant_tags * cant_opciones

def reemplazo(tag):
	if tag in cambios:
		return cambios[tag]
	else:
		return tag

def generate_cases(words):
	output = []
	largo = len(words)
	for i in range(largo):
		line = "["
		for j in range(largo):
			if j > 0:
				line += ","
			if "\"" in words[j][0]:
				line += "('" + words[j][0] + "'," + str(i - j) + ")"
			else:
				line += "(\"" + words[j][0] + "\"," + str(i - j) + ")"
		for j in range(largo,5):
			line += ",(\"OUT\"," + str(i - j) + ")"
		line += "] " + generate_vector_palabra(words[i], tags, opciones, largo_vector) + "\n"
		output.append(line)
	return output

def process_sentence(sentence_in):
	intermediate = []
	sentence = []
	for word in sentence_in:
		if " " in word[0]:
			words = word[0].split(" ")
			first = True
			for w in words:
				if w != "":
					aux_uno = number_filter(w)
					aux_dos = date_filter(aux_uno)
					aux_tres = correct_escape_sequences(aux_dos)
					if word[2] == "b":
						if first:
							sentence.append((aux_tres, word[1], "b"))
							first = False
						else:
							sentence.append((aux_tres, word[1], "i"))
					elif word[2] == "i":
						sentence.append((aux_tres, word[1], "i"))
					elif word[2] == "e":
						sentence.append((aux_tres, word[1], "i"))
					elif word[2] == "s":
						if first:
							sentence.append((aux_tres, word[1], "b"))
							first = False
						else:
							sentence.append((aux_tres, word[1], "i"))
			if word[2] == "e" or word[2] == "s":
				sentence[-1] = ((sentence[-1][0],sentence[-1][1],"e"))
		elif "_" in word[0]:
			words = word[0].split("_")
			first = True
			for w in words:
				if w != "":
					aux_uno = number_filter(w)
					aux_dos = date_filter(aux_uno)
					aux_tres = correct_escape_sequences(aux_dos)
					if word[2] == "b":
						if first:
							sentence.append((aux_tres, word[1], "b"))
							first = False
						else:
							sentence.append((aux_tres, word[1], "i"))
					elif word[2] == "i":
						sentence.append((aux_tres, word[1], "i"))
					elif word[2] == "e":
						sentence.append((aux_tres, word[1], "i"))
					elif word[2] == "s":
						if first:
							sentence.append((aux_tres, word[1], "b"))
							first = False
						else:
							sentence.append((aux_tres, word[1], "i"))
			if word[2] == "e" or word[2] == "s":
				sentence[-1] = ((sentence[-1][0],sentence[-1][1],"e"))
		elif word[0] != "":
			aux_uno = number_filter(word[0])
			aux_dos = date_filter(aux_uno)
			aux_tres = correct_escape_sequences(aux_dos)
			sentence.append((aux_tres,word[1],word[2]))
	length = len(sentence)
	hold = []
	for w in sentence:
		if w[1] == "":
			intermediate.append((w[0], None))
		elif w[2] == "b":
			for held in hold:
				intermediate.append((held[0],None))
			hold = [w]
		elif w[2] == "i":
			hold.append(w)
		elif w[2] == "e":
			for held in hold:
				intermediate.append(held)
			intermediate.append((w[0],w[1],"i"))
			hold = []
		elif w[2] == "s":
			for held in hold:
				intermediate.append((held[0],None))
			hold = []
			intermediate.append((w[0],w[1],"b"))
	output = generate_cases(intermediate)
	return output

def process_file(input_file, output_file):
	in_sentence = False
	sentence = []
	in_chunk = False
	first = "b"
	tag_seleccionado = ""
	for line in input_file:
		if not in_sentence and "<sentence" in line:
			in_sentence = True
			first = "b"
			in_chunk = False
			tag_seleccionado = ""
		if in_sentence and not in_chunk:
			if (" wd=\"") in line:
				palabra = re.sub(".* wd=\"","",line)
				palabra = re.sub("\".*\n","",palabra)
				sentence.append((palabra, "", first))
			else:
				for tag in tags:
					if (("<" + tag + ">") in line or ("<" + tag + " ") in line) and ("</" + tag + ">") not in line:
						in_chunk = True
						tag_seleccionado = tag
		elif in_sentence and in_chunk:
			if (" wd=\"") in line:
				palabra = re.sub(".* wd=\"","",line)
				palabra = re.sub("\".*\n","",palabra)
				palabra = palabra.decode('utf-8').lower().encode('utf-8')
				sentence.append((palabra,reemplazo(tag_seleccionado), first))
				if first == "b":
					first = "i"	
			if ("<" + tag_seleccionado) not in line and ("</" + tag_seleccionado + ">") in line:
				in_chunk = False
				first = "b"
				if sentence[-1][2] == "b" and reemplazo(tag_seleccionado) == sentence[-1][1]:
					sentence[-1] = (sentence[-1][0],sentence[-1][1],"s")
				elif sentence[-1][2] == "i" and reemplazo(tag_seleccionado) == sentence[-1][1]:
					sentence[-1] = (sentence[-1][0],sentence[-1][1],"e")
				tag_seleccionado = ""
			for tag in tags:
					if (("<" + tag + ">") in line or ("<" + tag + " ") in line) and ("</" + tag + ">") not in line:
						tag_seleccionado = tag
						first = "b"
		if in_sentence and "</sentence" in line:
			in_sentence = False
			output = process_sentence(sentence)
			for o in output:
				output_file.write(o)
			sentence = []

input_folder = sys.argv[1]
output_folder = sys.argv[2]

output_training_file = open(output_folder + "/" + "microchunking_training.csv","w")
output_testing_file = open(output_folder + "/" + "microchunking_testing.csv","w")
output_pruebas_file = open(output_folder + "/" + "microchunking_pruebas.csv","w")

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