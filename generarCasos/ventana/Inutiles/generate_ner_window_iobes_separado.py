import sys
import os
import re
import random
from funciones_generales import correct_escape_sequences, number_filter, date_filter, list_to_str
from funciones_vector import generate_vector_palabra_multiple, generate_vector_cero

person_token = 0
location_token = 1
organization_token = 2
other_token = 3
out_token = 4

tags = {"person" : 0, "location" : 1, "organization" : 2, "other" : 3}
opciones = {"b" : 0, "i" : 1, "e" : 2, "s" : 3}

in_tag = {}
for tag in tags:
	in_tag[tag] = 0

cant_opciones = len(opciones)
cant_tags = len(tags)
largo_vector = cant_tags + cant_opciones
window_size = int(11)

def generate_cases(words):
	output = []
	mitad_ventana = int(window_size / 2)
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
		if words[i][1] != None:
			line += generate_vector_palabra_multiple([words[i][1], words[i][2]], [tags, opciones], largo_vector) + "\n"
		else:
			line += list_to_str(generate_vector_cero(largo_vector)) + "\n"
		output.append(line)
	return output

def process_sentence(sentence_in):
	intermediate = []
	sentence = []
	for word in sentence_in:
		if " " in word[0]:
			words = word[0].split(" ")
			for w in words:
				if w != "":
					aux_uno = number_filter(w)
					aux_dos = date_filter(aux_uno)
					aux_tres = correct_escape_sequences(aux_dos)
					sentence.append((aux_tres, word[1]))
		elif "_" in word[0]:
			words = word[0].split("_")
			for w in words:
				if w != "":
					aux_uno = number_filter(w)
					aux_dos = date_filter(aux_uno)
					aux_tres = correct_escape_sequences(aux_dos)
					sentence.append((aux_tres,word[1]))
		elif word[0] != "":
			aux_uno = number_filter(word[0])
			aux_dos = date_filter(aux_uno)
			aux_tres = correct_escape_sequences(aux_dos)
			sentence.append((aux_tres,word[1]))
	length = len(sentence)
	in_person = False
	in_location = False
	in_organization = False
	in_other = False
	for i in range(length):
		if sentence[i][1] == None:
			intermediate.append((sentence[i][0],None))
		elif sentence[i][1] == "person":
			if not in_person and i < (len(sentence) - 1) and sentence[i + 1][1] == "person":
				in_person = True
				intermediate.append((sentence[i][0],"person","b"))
			elif not in_person:
				intermediate.append((sentence[i][0],"person","s"))
			elif in_person and i < (len(sentence) - 1) and sentence[i + 1][1] == "person":
				intermediate.append((sentence[i][0],"person","i"))
			elif in_person:
				intermediate.append((sentence[i][0],"person","e"))
				in_person = False
		elif sentence[i][1] == "location":
			if not in_location and i < (len(sentence) - 1) and sentence[i + 1][1] == "location":
				in_location = True
				intermediate.append((sentence[i][0],"location","b"))
			elif not in_location:
				intermediate.append((sentence[i][0],"location","s"))
			elif in_location and i < (len(sentence) - 1) and sentence[i + 1][1] == "location":
				intermediate.append((sentence[i][0],"location","i"))
			elif in_location:
				intermediate.append((sentence[i][0],"location","e"))
				in_location = False
		elif sentence[i][1] == "organization":
			if not in_organization and i < (len(sentence) - 1) and sentence[i + 1][1] == "organization":
				in_organization = True
				intermediate.append((sentence[i][0],"organization","b"))
			elif not in_organization:
				intermediate.append((sentence[i][0],"organization","s"))
			elif in_organization and i < (len(sentence) - 1) and sentence[i + 1][1] == "organization":
				intermediate.append((sentence[i][0],"organization","i"))
			elif in_organization:
				intermediate.append((sentence[i][0],"organization","e"))
				in_organization = False
		elif sentence[i][1] == "other":
			if not in_other and i < (len(sentence) - 1) and sentence[i + 1][1] == "other":
				in_other = True
				intermediate.append((sentence[i][0],"other","b"))
			elif not in_other:
				intermediate.append((sentence[i][0],"other","s"))
			elif in_other and i < (len(sentence) - 1) and sentence[i + 1][1] == "other":
				intermediate.append((sentence[i][0],"other","i"))
			elif in_other:
				intermediate.append((sentence[i][0],"other","e"))
				in_other = False
	output = generate_cases(intermediate)
	return output

def process_file(input_file, output_file):
	in_sentence = False
	sentence = []
	for line in input_file:
		if not in_sentence and "<sentence" in line:
			in_sentence = True
		if in_sentence and " wd=" in line and " ne=\"person\"" in line:
			aux_line = re.sub(".*?wd=\"","",line)
			word = re.sub("\".*\n","",aux_line)
			sentence.append((word,"person"))
		elif in_sentence and " wd=" in line and " ne=\"location\"" in line:
			aux_line = re.sub(".*?wd=\"","",line)
			word = re.sub("\".*\n","","location")
			sentence.append((word,location_token))
		elif in_sentence and " wd=" in line and " ne=\"organization\"" in line:
			aux_line = re.sub(".*?wd=\"","",line)
			word = re.sub("\".*\n","",aux_line)
			sentence.append((word,"organization"))
		elif in_sentence and " wd=" in line and " ne=\"other\"" in line:
			aux_line = re.sub(".*?wd=\"","",line)
			word = re.sub("\".*\n","",aux_line)
			sentence.append((word,"other"))
		elif in_sentence and " wd=" in line:
			aux_line = re.sub(".*?wd=\"","",line)
			word = re.sub("\".*\n","",aux_line)
			sentence.append((word,None))
		if in_sentence and "</sentence" in line:
			in_sentence = False
			output = process_sentence(sentence)
			for o in output:
				output_file.write(o)
			sentence = []

input_folder = sys.argv[1]
output_folder = sys.argv[2]

output_training_file = open(output_folder + "/" + "ner_training_iobes_separado.csv","w")
output_testing_file = open(output_folder + "/" + "ner_testing_iobes_separado.csv","w")
output_pruebas_file = open(output_folder + "/" + "ner_pruebas_iobes_separado.csv","w")

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
