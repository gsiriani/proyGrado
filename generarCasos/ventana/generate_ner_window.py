import sys
import os
import re
import random
from funciones_generales import correct_escape_sequences, number_filter, date_filter

person_token = 0
location_token = 1
organization_token = 2
other_token = 3
out_token = 4
out = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
b_per = "1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
i_per = "0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
e_per = "0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0"
s_per = "0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0"
b_loc = "0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0"
i_loc = "0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0"
e_loc = "0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0"
s_loc = "0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0"
b_org = "0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0"
i_org = "0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0"
e_org = "0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0"
s_org = "0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0"
b_oth = "0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0"
i_oth = "0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0"
e_oth = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0"
s_oth = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1"

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
		word[0] = word[0].decode('utf-8').lower().encode('utf-8')
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
		if sentence[i][1] == out_token:
			intermediate.append((sentence[i][0],out))
		elif sentence[i][1] == person_token:
			if not in_person and i < (len(sentence) - 1) and sentence[i + 1][1] == person_token:
				in_person = True
				intermediate.append((sentence[i][0],b_per))
			elif not in_person:
				intermediate.append((sentence[i][0],s_per))
			elif in_person and i < (len(sentence) - 1) and sentence[i + 1][1] == person_token:
				intermediate.append((sentence[i][0],i_per))
			elif in_person:
				intermediate.append((sentence[i][0],e_per))
				in_person = False
		elif sentence[i][1] == location_token:
			if not in_location and i < (len(sentence) - 1) and sentence[i + 1][1] == location_token:
				in_location = True
				intermediate.append((sentence[i][0],b_loc))
			elif not in_location:
				intermediate.append((sentence[i][0],s_loc))
			elif in_location and i < (len(sentence) - 1) and sentence[i + 1][1] == location_token:
				intermediate.append((sentence[i][0],i_loc))
			elif in_location:
				intermediate.append((sentence[i][0],e_loc))
				in_location = False
		elif sentence[i][1] == organization_token:
			if not in_organization and i < (len(sentence) - 1) and sentence[i + 1][1] == organization_token:
				in_organization = True
				intermediate.append((sentence[i][0],b_org))
			elif not in_organization:
				intermediate.append((sentence[i][0],s_org))
			elif in_organization and i < (len(sentence) - 1) and sentence[i + 1][1] == organization_token:
				intermediate.append((sentence[i][0],i_org))
			elif in_organization:
				intermediate.append((sentence[i][0],e_org))
				in_organization = False
		elif sentence[i][1] == other_token:
			if not in_other and i < (len(sentence) - 1) and sentence[i + 1][1] == other_token:
				in_other = True
				intermediate.append((sentence[i][0],b_oth))
			elif not in_other:
				intermediate.append((sentence[i][0],s_oth))
			elif in_other and i < (len(sentence) - 1) and sentence[i + 1][1] == other_token:
				intermediate.append((sentence[i][0],i_oth))
			elif in_other:
				intermediate.append((sentence[i][0],e_oth))
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
			sentence.append((word,person_token))
		elif in_sentence and " wd=" in line and " ne=\"location\"" in line:
			aux_line = re.sub(".*?wd=\"","",line)
			word = re.sub("\".*\n","",aux_line)
			sentence.append((word,location_token))
		elif in_sentence and " wd=" in line and " ne=\"organization\"" in line:
			aux_line = re.sub(".*?wd=\"","",line)
			word = re.sub("\".*\n","",aux_line)
			sentence.append((word,organization_token))
		elif in_sentence and " wd=" in line and " ne=\"other\"" in line:
			aux_line = re.sub(".*?wd=\"","",line)
			word = re.sub("\".*\n","",aux_line)
			sentence.append((word,other_token))
		elif in_sentence and " wd=" in line:
			aux_line = re.sub(".*?wd=\"","",line)
			word = re.sub("\".*\n","",aux_line)
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

output_training_file = open(output_folder + "/" + "ner_training.csv","w")
output_testing_file = open(output_folder + "/" + "ner_testing.csv","w")
output_pruebas_file = open(output_folder + "/" + "ner_pruebas.csv","w")

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
