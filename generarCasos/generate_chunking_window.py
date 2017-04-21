import sys
import os
import re
import random

tags = {"sn" : 0, "sa" : 1, "s.a" : 2, "sp" : 3, "sadv" : 4, "grup.verb": 5}
opciones = {"b" : 0, "i" : 1, "e" : 2, "s" : 3}

in_tag = {}
for tag in tags:
	in_tag[tag] = 0

cant_opciones = len(opciones)
cant_tags = len(tags)
largo_vector = cant_tags * cant_opciones

def obtener_indice(tag, opcion):
	indice = tags[tag] + opciones[opcion]
	return indice

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

def vector_variante(indice):
	vector = []
	for i in range(cant_tags):
		if i == indice:
			vector.append(1)
		else:
			vector.append(0)
	return vector

def correct_escape_sequences(word):
	if word == "&quot;":
		return "\""
  	elif word == "&lt;":
  		return "<"
	elif word == "&gt;":
		return ">"
	elif word == "&amp;":
		return "&"
	else:
		return word

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

def generate_cases(words):
	output = []
	for i in range(len(words)):
		line = ""
		max_index = min(i + 6,len(words))
		min_index = max(0,i - 5)
		for j in range(0,5 - i):
			line += "OUT "
		for j in range(min_index, max_index):
			line += words[j][0] + " "
		for j in range(6 - (len(words) - i)):
			line += "OUT "
		line += words[i][1] + "\n"
		output.append(line)
	return output

def process_sentence(sentence_in):
	intermediate = []
	sentence = []
	for word in sentence_in:
		if "_" in word[0]:
			words = word[0].split("_")
			for w in words:
				aux_uno = number_filter(w)
				aux_dos = date_filter(aux_uno)
				aux_tres = correct_escape_sequences(aux_dos)
				sentence.append((aux_tres,word[1]))
		else:
			aux_uno = number_filter(word[0])
			aux_dos = date_filter(aux_uno)
			aux_tres = correct_escape_sequences(aux_dos)
			sentence.append((aux_tres,word[1]))
	length = len(sentence)
	in_sn = False
	in_sv = False
	for i in range(length):
		if sentence[i][1] == out_token:
			intermediate.append((sentence[i][0],out))
		if sentence[i][1] == sn_token:
			if not in_sn and i < (len(sentence) - 1) and sentence[i + 1][1] == sn_token:
				in_sn = True
				intermediate.append((sentence[i][0],b_sn))
			elif not in_sn:
				intermediate.append((sentence[i][0],s_sn))
			elif in_sn and i < (len(sentence) - 1) and sentence[i + 1][1] == sn_token:
				intermediate.append((sentence[i][0],i_sn))
			elif in_sn:
				intermediate.append((sentence[i][0],e_sn))
				in_sn = False
		elif sentence[i][1] == sv_token:
			if not in_sv and i < (len(sentence) - 1) and sentence[i + 1][1] == sv_token:
				in_sv = True
				intermediate.append((sentence[i][0],b_sv))
			elif not in_sv:
				intermediate.append((sentence[i][0],s_sv))
			elif in_sv and i < (len(sentence) - 1) and sentence[i + 1][1] == sv_token:
				intermediate.append((sentence[i][0],i_sv))
			elif in_sv:
				intermediate.append((sentence[i][0],e_sv))
				in_sv = False
	output = generate_cases(intermediate)
	return output

def generate_in_chunk():
	l = []
	for i in range(len(tags)):
		l.append(0)
	return l

def process_file(input_file, output_file):
	in_sentence = False
	sn = 0
	sv = 0
	sentence = []
	in_chunk = []
	for line in input_file:
		if not in_sentence and "<sentence" in line:
			in_sentence = True
			in_chunk = generate_in_chunk()
		if in_sentence and all(map(lambda x : x == 0, in_chunk)):
			for tag in tags:
				if ("<" + tag) in line and ("</" + tag + ">") not in line:
					in_chunk[tags[tag]] += 1
				elif (" wd=\"") in line:
					palabra = re.sub(".* wd=\"","",line)
					palabra = re.sub("\".*\n","",palabra)
					sentence.append((palabra,None,None))




					
		if in_sentence and sv == 0 and "<sn" in line:
			sn += 1
		if in_sentence and sn == 0 and "<grup.verb" in line:
			sv += 1
		if in_sentence and sn > 0 and "</sn>" in line:
			sn -= 1
		if in_sentence and sv > 0 and "</grup.verb>":
			sv -= 1
		if in_sentence and "wd=" in line:
			aux_line = re.sub(".*?wd=\"","",line)
			word = re.sub("\".*\n","",aux_line)
			if sn > 0:
				sentence.append((word,sn_token))
			elif sv > 0:
				sentence.append((word,sv_token))
			else:
				sentence.append((word,out_token))
		if in_sentence and "</sentence" in line:
			in_sentence = False
			sn = 0
			sv = 0
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