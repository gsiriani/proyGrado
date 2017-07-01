#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import re
import random
import freeling
from funciones_generales import list_to_str, correct_escape_sequences, number_filter, date_filter, list_to_str_utf8
from funciones_vector import vector_variante

tamanio_ventana = 11
# Orden de los tags en aparicion
orden_tags = {"pos" : 0}

# Cantidad de opciones por tag segun el orden
cantidad_opciones = [12]

# Opciones para cada tag
opciones_pos = {"a" : 0,
				"c" : 1,
				"d":  2,
				"f" : 3,
				"i" : 4,
				"n" : 5,
				"p" : 6,
				"r" : 7,
				"s" : 8,
				"v" : 9,
				"w" : 10,
				"z" : 11}


# Diccionario de opciones de tag
opciones_tags = {"pos" : opciones_pos}

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
		line += list_to_str(words[i][1]) + "\n"
		output.append(line)
	return output

def process_freeling(sentence_in, freeling_list):
	sentence = []
	freeling_sentence = []
	salida = []
	for word in sentence_in:
		sentence.append(word[0])
	oracion = list_to_str_utf8(sentence)
	#print oracion
	#print "chk1"
	l = freeling_list[1].tokenize(oracion)
	#print "chk2"
	ls = freeling_list[2].split(freeling_list[0],l,False)
	#print "chk3"
	ls = freeling_list[3].analyze(ls)
	#print "chk4"
	ls = freeling_list[4].analyze(ls)
	#print "chk5"
	ls = freeling_list[5].analyze(ls)
	#print "chk6"
	ls = freeling_list[6].analyze(ls)
	#print "chk7"
	ls = freeling_list[7].analyze(ls)
	#print "chk8"
	for s in ls:
		ws = s.get_words()
		for w in ws:
			freeling_sentence.append((w.get_form(),w.get_tag().lower()[0]))
	for i in sentence_in:
		print i[0] + " " + str(i[2])
		if i[2] == 1:
			estaba = False
			for k in freeling_sentence:
				if i[0] == k[0]:
					salida.append((k[0],k[1]))
					estaba = True
			if not estaba:
				print "Error de clave"
				return sentence_in
		else:
			salida.append((i[0],i[1]))
	return salida

def process_sentence(sentence_in, freeling_list):
	sentence = []
	freeling = False
	intermediate = []
	for word in sentence_in:
		if " " in word[0]:
			words = word[0].split(" ")
			for w in words:
				if w != "":
					sentence.append((w, word[1], 1))
			freeling = True
		elif "_" in word[0]:
			words = word[0].split("_")
			for w in words:
				if w != "":
					sentence.append((w, word[1], 1))
			freeling = True
		else:
			sentence.append((word[0], word[1], 0))
	if freeling:
		sentence = process_freeling(sentence, freeling_list)
	for word in sentence:
		aux_uno = number_filter(word[0])
		aux_dos = date_filter(aux_uno)
		aux_tres = correct_escape_sequences(aux_dos)
		pos_tag = vector_variante(opciones_pos[word[1]],tamanio_ventana)
		intermediate.append((aux_tres,pos_tag))
	output = generate_cases(intermediate)
	return output

def process_file(input_file, output_file, freeling_list):
	in_sentence = False
	sentence = []
	for line_e in input_file:
		line = line_e.decode("utf-8")
		if not in_sentence and "<sentence" in line:
			in_sentence = True
		if in_sentence and " wd=" in line:
			aux_word = re.sub(".*?wd=\"","",line)
			word = re.sub("\".*\n","",aux_word)
			pos = re.sub(".*?<","",line)[0]
			sentence.append((word,pos))
		if in_sentence and "</sentence" in line:
			in_sentence = False
			output = process_sentence(sentence, freeling_list)
			for o in output:
				print o
				output_file.write(o)
			sentence = []

input_folder = sys.argv[1]
salida_folder = sys.argv[2]

output_training_file = open(salida_folder + "/pos_simple_training.csv","w")
output_testing_file = open(salida_folder + "/pos_simple_testing.csv","w")
output_pruebas_file = open(salida_folder + "/pos_simple_pruebas.csv","w")

input_training_file = open(input_folder + "/" + "ancora_training.xml","r")
input_testing_file = open(input_folder + "/" + "ancora_testing.xml","r")
input_pruebas_file = open(input_folder + "/" + "ancora_pruebas.xml","r")

FREELINGDIR = "/usr/local";

DATA = FREELINGDIR+"/share/freeling/";
LANG="es";

freeling.util_init_locale("default");

op= freeling.maco_options("es");
op.set_data_files( "",
                   DATA + "common/punct.dat",
                   DATA + LANG + "/dicc.src",
                   DATA + LANG + "/afixos.dat",
                   "",
                   DATA + LANG + "/locucions.dat", 
                   DATA + LANG + "/np.dat",
                   DATA + LANG + "/quantities.dat",
                   DATA + LANG + "/probabilitats.dat");

tk=freeling.tokenizer(DATA+LANG+"/tokenizer.dat");
sp=freeling.splitter(DATA+LANG+"/splitter.dat");
sid=sp.open_session();
mf=freeling.maco(op);

mf.set_active_options(False, True, True, True,
                      True, True, False, True,
                      True, True, True, True )

tg=freeling.hmm_tagger(DATA+LANG+"/tagger.dat",True,2)
sen=freeling.senses(DATA+LANG+"/senses.dat")
parser= freeling.chart_parser(DATA+LANG+"/chunker/grammar-chunk.dat")
dep=freeling.dep_txala(DATA+LANG+"/dep_txala/dependences.dat", parser.get_start_symbol())


process_file(input_training_file, output_training_file, [sid, tk, sp, mf, tg, sen, parser, dep])
process_file(input_testing_file, output_testing_file, [sid, tk, sp, mf, tg, sen, parser, dep])
process_file(input_pruebas_file, output_pruebas_file, [sid, tk, sp, mf, tg, sen, parser, dep])

input_training_file.close()
input_pruebas_file.close()
input_testing_file.close()

output_pruebas_file.close()
output_testing_file.close()
output_training_file.close()
    
sp.close_session(sid);