import sys
import os
import re
import random
from funciones_generales import correct_escape_sequences, number_filter, date_filter
from funciones_vector import generate_vector_cero, generate_vector_palabra

opciones_arg = ["arg0", "arg1", "arg2", "arg3", "arg4", "argL", "argM", "verb"]
opciones_iobes = ["b", "i", "e", "s"]

input_folder = sys.argv[1]
output_folder = sys.argv[2]

tag_length = len(opciones_iobes) * len(opciones_arg)
out_tag = []
for i in range(tag_length):
	out_tag.append(0)

cant_opciones = len(opciones)
cant_tags = len(tags)
largo_vector = cant_tags * cant_opciones

def generate_cases(words):
	output = []
	largo = len(words)
	for i in range(largo):
		line = str(largo) + " "
		indices = ""
		for j in range(len(words)):
			line += words[j][0] + " "
			indices += str(i - j) + " "
		line += indices + " " + generate_vector_palabra(words[i], tags, opciones, largo_vector) + "\n"
		output.append(line)
	return output

def process_sentence(sentence_in):
	intermediate = []
	sentence = []
	for word in sentence_in:
		if "_" in word[0]:
			words = word[0].split("_")
			primero = True
			for w in words:
				aux_uno = number_filter(w)
				aux_dos = date_filter(aux_uno)
				aux_tres = correct_escape_sequences(aux_dos)
				if primero:
					sentence.append((aux_tres, word[1], word[2]))
				else:
					sentence.append((aux_tres, word[1], False))
		else:
			aux_uno = number_filter(word[0])
			aux_dos = date_filter(aux_uno)
			aux_tres = correct_escape_sequences(aux_dos)
			sentence.append((aux_tres, word[1], word[2]))
	length = len(sentence)
	for i in range(length):
		if sentence[i][1] == None:
			intermediate.append((sentence[i][0], None))
		elif sentence[i][2]:
			if i < (len(sentence) - 1) and not sentence[i + 1][2]:
				intermediate.append((sentence[i][0], sentence[i][1], "b"))
			else:
				intermediate.append((sentence[i][0], sentence[i][1], "s"))
		else:
			if i < (len(sentence) - 1) and not sentence[i + 1][2]:
				intermediate.append((sentence[i][0], sentence[i][1], "i"))
			else:
				intermediate.append((sentence[i][0], sentence[i][1], "e"))
	output = generate_cases(intermediate)
	return output

def process_sentence_iterativo(sentence_in):
	i = 0
	end = False
	output = []
	while not end:
		in_arg = False
		in_verb = False
		first = True
		end = True
		arg = ""
		sentence = []
		j = 0
		for line in sentence_in:
			if j == i:
				end = False
			if j == i and re.match(".* arg=\".*"):
				in_arg = True
				arg = re.sub(".* arg=\"", "", line)
				arg = re.sub("\".*\n", "", arg)
				first = True
			if j == i and re.match("<grup.verb"):
				in_verb
				first = True
			if re.match(".*<.*>.*", line) and not re.match(".*<.*/>.*", line):
				j += 1
			if re.match(".*</.*>.*", line):
				j -= 1
				if j == i:
					in_arg = False
					in_verb = False
			if re.match(".* wd=\"", line):
				word = re.sub(".* wd=\"", "", line)
				word = re.sub("\".*\n", "", word)
				if in_arg:
					sentence.append(word, arg, first)
					first = False
				elif in_verb:
					sentence.append(word, "verb", first)
					first = False
				else:
					sentence.append(word, "out", True)
		output += process_sentence(sentence)
		i += 1
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
			output = process_sentence_iterativo(sentence)
			for o in output:
				output_file.write(o)
			sentence = []
		if in_sentence:
			sentence.append(line)

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