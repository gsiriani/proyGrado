import sys
import os
import re
import random
from funciones_generales import correct_escape_sequences, number_filter, date_filter

separador = ","
default = "UNK"

def process_sentence(sentence_in, lexicon_dicc):
	salida = ""
	sentence = []
	for word in sentence_in:
		if " " in word:
			words = word.split(" ")
			for w in words:
				if w != "":
					aux_uno = number_filter(w)
					aux_dos = date_filter(aux_uno)
					aux_tres = correct_escape_sequences(aux_dos)
					sentence.append(aux_tres)
		elif "_" in word:
			words = word.split("_")
			for w in words:
				if w != "":
					aux_uno = number_filter(w)
					aux_dos = date_filter(aux_uno)
					aux_tres = correct_escape_sequences(aux_dos)
					sentence.append(aux_tres)
		elif word != "":
			aux_uno = number_filter(word)
			aux_dos = date_filter(aux_uno)
			aux_tres = correct_escape_sequences(aux_dos)
			sentence.append(aux_tres)
	for i in sentence:
		if salida == "":
			salida = lexicon_dicc.setdefault(i, lexicon_dicc[default])
		else:
			salida += "," + lexicon_dicc.setdefault(i, lexicon_dicc[default])
	salida += "\n"
	return salida

def process_file(input_file, output_file, lexicon_dicc):
	in_sentence = False
	sentence = []
	for line in input_file:
		if not in_sentence and "<sentence" in line:
			in_sentence = True
		if in_sentence and " wd=" in line:
			aux_line = re.sub(".*?wd=\"","",line)
			word = re.sub("\".*\n","",aux_line)
			sentence.append(word)
		if in_sentence and "</sentence" in line:
			in_sentence = False
			output = process_sentence(sentence, lexicon_dicc)
			for o in output:
				output_file.write(o)
			sentence = []

lexicon_file = open(sys.argv[1],"r")
input_folder = sys.argv[2]
output_folder = sys.argv[3]

output_training_file = open(output_folder + "/" + "oraciones_training.csv","w")
output_testing_file = open(output_folder + "/" + "oraciones_testing.csv","w")
output_pruebas_file = open(output_folder + "/" + "oraciones_pruebas.csv","w")

input_training_file = open(input_folder + "/" + "ancora_training.xml","r")
input_testing_file = open(input_folder + "/" + "ancora_testing.xml","r")
input_pruebas_file = open(input_folder + "/" + "ancora_pruebas.xml","r")

lexicon_dicc = {}
i = 0
for line in lexicon_file:
	lexicon_dicc[line.replace("\n","")] = str(i)
	i += 1

process_file(input_training_file, output_training_file, lexicon_dicc)
process_file(input_testing_file, output_testing_file, lexicon_dicc)
process_file(input_pruebas_file, output_pruebas_file, lexicon_dicc)

input_training_file.close()
input_pruebas_file.close()
input_testing_file.close()

output_pruebas_file.close()
output_testing_file.close()
output_training_file.close()