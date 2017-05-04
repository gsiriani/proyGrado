import sys
import os
import re
import random
from funciones_generales import correct_escape_sequences, number_filter, date_filter
from funciones_vector import generate_vector_cero, generate_vector_palabra

# Orden de aparicion de func
orden_func = {"ao" = 0,
			  "atr" = 1,
			  "cag" = 2,
			  "cc" = 3,
			  "cd" = 4,
			  "ci" = 5,
			  "cpred" = 6,
			  "creg" = 7,
			  "et" = 8,
			  "impers" = 9,
			  "mod" = 10,
			  "pass" = 11,
			  "suj" = 12}

# Orden de los args en aparicion
orden_arg = {"arg0" : 0,
			 "arg1" : 1,
			 "arg2" : 2,
			 "arg3" : 3,
			 "arg4" : 4,
			 "argl" : 5,
			 "argm" : 6,
			 "verb" : 7}

# Orden de los temas en aparicion
orden_tem = {"adv" = 0,
			 "agt" = 1,
			 "atr" = 2,
			 "ben" = 3,
			 "cau" = 4,
			 "cot" = 5,
			 "des" = 6,
			 "efi" = 7,
			 "ein" = 8,
			 "exp" = 9,
			 "ext" = 10,
			 "fin" = 11,
			 "ins" = 12,
			 "loc" = 13,
			 "mnr" = 14,
			 "ori" = 15,
			 "pat" = 16,
			 "src" = 17,
			 "tem" = 18,
			 "tmp" = 19}

# Opciones de verbo
opciones_verbo = {"normal" = 0,
				  "infinitivo" = 1,
				  "nombre_deverbal" = 2}

# Opciones lss de verbo
opciones_lss = {"a11.transitive-causative" = 0,
				"a12.ditransitive-causative-state" = 1,
				"a13.ditransitive-causative-instrumental" = 2,
				"a21.transitive-agentive-patient" = 3,
				"a22.transitive-agentive-theme" = 4,
				"a23.transitive-agentive-extension" = 5,
				"a31.ditransitive-patient-locative" = 6,
				"a32.ditransitive-patient-benefactive" = 7,
				"a33.ditransitive-theme-locative" = 8,
				"a34.ditransitive-patient-theme" = 9,
				"a35.ditransitive-theme-cotheme" = 10,
				"b11.unaccusative-motion" = 11,
				"b12.unaccusative-passive-ditransitive" = 12,
				"b21.unaccusative-state" = 13,
				"b22.unaccusative-passive-transitive" = 14,
				"b23.unaccusative-theme-cotheme" = 15,
				"c11.state-existential" = 16,
				"c21.state-attributive" = 17,
				"c31.state-scalar" = 18,
				"c41.state-benefactive" = 19,
				"c42.state-experiencer" = 20,
				"d11.inergative-agentive" = 21,
				"d21.inergative-experiencer" = 22,
				"d31.inergative-source" = 23}

# Cantidad de opciones por tag segun el orden
cantidad_opciones = [13, 8, 20, 3, 24]



opciones_iobes = {"b" : 0, "i" : 1, "e" : 2, "s" : 3}

input_folder = sys.argv[1]
output_folder = sys.argv[2]

cant_opciones = len(opciones_iobes)
cant_tags = len(opciones_arg)
largo_vector = cant_tags * cant_opciones

def generate_cases(words, indice_verbo):
	output = []
	largo = len(words)
	for i in range(largo):
		line = "["
		for j in range(len(words)):
			if j > 0:
				line += ","
			if "\"" in words[j][0]:
				line += "('" + words[j][0] + "'," + str(i - j) + "," + str(indice_verbo - j) + ")"
			else:
				line += "(\"" + words[j][0] + "\"," + str(i - j) + "," + str(indice_verbo - j) + ")"
		for j in range(largo,5):
			line += ",(\"OUT\"," + str(i - j) + ")"
		line += "] " + generate_vector_palabra(words[i], opciones_arg, opciones_iobes, largo_vector) + "\n"
		output.append(line)
	return output

def process_sentence(sentence_in, indice_verbo):
	intermediate = []
	sentence = []
	for word in sentence_in:
		if " " in word[0]:
			words = word[0].split(" ")
			first = True
			for w in words:
				aux_uno = number_filter(w)
				aux_dos = date_filter(aux_uno)
				aux_tres = correct_escape_sequences(aux_dos)
				if first:
					sentence.append((aux_tres, word[1], word[2]))
					first = False
				else:
					sentence.append((aux_tres, word[1], False))
		elif "_" in word[0]:
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
	output = generate_cases(intermediate, indice_verbo)
	return output

def process_sentence_iterativo(sentence_in, sectores):
	output = []
	for sector in sectores:
		found = False
		in_arg = False
		in_verb = False
		first = True
		end = True
		arg = ""
		sentence = []
		indice_verbo = -1
		j = 0
		k = 0
		for line in sentence_in:
			if j == sector[2] and re.match(".* arg=\".*", line) and k >= sector[0] and k <= sector[1]:
				found = True
				in_arg = True
				arg = re.sub(".* arg=\"", "", line)
				arg = re.sub("\".*\n", "", arg)
				first = True
			if j == sector[2] and (re.match(".*<grup.verb.*", line) or re.match(".*<infinitiu.*", line)) and k >= sector[0] and k <= sector[1]:
				found = True
				indice_verbo = len(sentence)
				in_verb = True
				first = True					
			if re.match(".*<.*>.*", line) and not re.match(".*<.*/.*>.*", line):
				j += 1
			if re.match(".*</.*>.*", line):
				j -= 1
				if j == sector[2]:
					in_arg = False
					in_verb = False
			if re.match(".* wd=\"", line):
				word = re.sub(".* wd=\"", "", line)
				word = re.sub("\".*\n", "", word)
				if in_arg:
					sentence.append((word, arg, first))
					first = False
				elif in_verb:
					sentence.append((word, "verb", first))
					first = False
				elif sector[2] == j and re.match(".*<n .*origin=\"deverbal\".*", line) and k >= sector[0] and k <= sector[1]:
					found = True
					indice_verbo = len(sentence)
					sentence.append((word, "verb", first))
				else:
					sentence.append((word, None, True))
			k += 1
		if found and indice_verbo != -1:
			output += process_sentence(sentence, indice_verbo)
	return output

def determinar_sectores(sentence_in):
	i = 0
	end = False
	sectores = []
	while not end:
		j = 0
		end = True
		k = 0
		ini_sector = 0
		for line in sentence_in:
			if j == i:
				end = False
			if re.match(".*<.*>.*", line) and not re.match(".*<.*/.*>.*", line):
				j += 1
				if j == i + 1:
					ini_sector = k + 1
			if re.match(".*</.*>.*", line):
				j -= 1
				if j == i:
					sectores.append((ini_sector,k - 1, i + 1))
			k += 1
		i += 1
	return sectores

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
			sectores = determinar_sectores(sentence)
			output = process_sentence_iterativo(sentence, sectores)
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