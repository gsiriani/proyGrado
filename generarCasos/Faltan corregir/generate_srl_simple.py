import sys
import os
import re
import random
from funciones_generales import correct_escape_sequences, number_filter, date_filter

opciones_arg = ["arg0", "arg1", "arg2", "arg3", "arg4", "argL", "argM", "grup.verb"]
opciones_iobes = ["b", "i", "e", "s"]

input_folder = sys.argv[1]
output_folder = sys.argv[2]

tag_length = len(opciones_iobes) * len(opciones_arg)
out_tag = []
for i in range(tag_length):
	out_tag.append(0)

def generate_cases(words):
	output = []
	largo = len(words)
	for i in range(largo):
		line = str(largo) + " "
		indices = ""
		for j in range(len(words)):
			line += words[j][0] + " "
			indices += str(i - j) + " "
		line += indices + words[i][1] + "\n"
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

def process_file(input_file, output_file):
	in_sentence = False
	sn = 0
	sv = 0
	sentence = []
	for line in input_file:
		if not in_sentence and "<sentence" in line:
			in_sentence = True
		if "</sentence" in line:
			in_sentence = False
			output = process_sentence(sentence)
			for o in output:
				output_file.write(o)
			sentence = []
		if in_sentence:
			sentence.append(line)

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

output_training_file = open(output_folder + "/" + "srl_simple_training.csv","w")
output_testing_file = open(output_folder + "/" + "srl_simple_testing.csv","w")
output_pruebas_file = open(output_folder + "/" + "srl_simple_pruebas.csv","w")

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