#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import re
import random
import freeling
import unicodedata
from funciones_generales import list_to_str, correct_escape_sequences, number_filter, date_filter, list_to_str_utf8
from funciones_vector import vector_variante

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

largo_vector = len(opciones_pos)

lista_separables = {"del" : 2, "al" : 2, "¡qué" : 2, "rajada!" : 2, "¡agua" : 2, "va!" : 2, "`toro`" : 3,
					"\"el" : 2, "pibe\"" : 2, ",obligado" : 2, "'savoir" : 2, "faire'" : 2, "(capital" : 2, "carintia)" : 2} #, "1993," : 2, "g-77," : 2, "arte," : 2, "baby," : 2}
# "\"steel\"" : 3, "\"guga\"" : 3, , "\"tucho\"" : 3
#resto = {"f." : 2, "w." : 2, "c." : 2, "l." : 2, "del'" : 3, "p." : 2, "manaos" : 2, "lamela" : 2, "s." : 2, "sete" : 2,
		 #"haciéndose" : 2, "m." : 2, "som!" : 2, "u.d." : 4, "n." : 2, "b." : 2, "d." : 2, "surfin'" : 2, "darse" : 2}
ancora = {"u.d." : 4, "del'" : 3, "manaos" : 2, "lamela" : 2, "sete" : 2, "paral.leles" : 3, "cavale" : 2, "ponte" : 2,
		  "-fpa-" : 3, "-fpt-" : 3, "cascales" : 2, "a." : 2, "g.p." : 4, "angeles'84" : 3, "atrevéos" : 2, "e." : 2, "2°" : 2,
		  "sub`21" : 3, "`cash" : 2, "`robert" : 2, "o`higgins" : 3, "o`neal" : 3, "portela" : 2, "cash`s" : 3, "perestelo" : 2,
		  "parente" : 2, "mudela" : 2, "dale" : 2, "s.l." : 4, "c.r." : 4, ",comerç" : 2, "st.louis" : 3, "señalándole" : 2,
		  "-lsb-" : 3, "o`neill" : 3, "empeñándose" :2, "d´ebre" : 3, "merelo" : 2, "winnipeg'99" : 3, "francia`98" : 3, "b.m." : 4,
		  "mandela" : 2, "o`brien" : 3, "adela" : 2}

free = {"salvarse" : 2, "suicídate" : 2, "písale" : 2}

# Diccionario de opciones de tag
opciones_tags = {"pos" : opciones_pos}

def palabra_citada(word):
	return re.match("\".*\"",word)

def quitar_tildes(word):
	return ''.join((c for c in unicodedata.normalize('NFD', word) if unicodedata.category(c) != 'Mn'))

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
		line += "] " + list_to_str(words[i][1]) + "\n"
		output.append(line)
	return output

def es_palabra_con_coma(palabra, palabra_sep, coma):
	return coma == "," and palabra == palabra_sep + coma

def es_palabra_compuesta(palabra_o, palabra1, palabra2, valor_o, valor1):
	p_iguales = palabra_o == palabra1 + palabra2 or quitar_tildes(palabra_o) == palabra1 + palabra2
	v_iguales = valor_o == valor1
	return p_iguales and v_iguales

def process_freeling(sentence_in, freeling_list):
	sentence = []
	freeling_sentence = []
	salida = []
	for word in sentence_in:
		sentence.append(word[0])
	oracion = list_to_str_utf8(sentence)
	#print oracion
	if not re.match(".*\.$", oracion) or re.match(".*\.\.\.$", oracion):
		oracion += " ."
	sid = freeling_list[2].open_session()
	l = freeling_list[1].tokenize(oracion)
	ls = freeling_list[2].split(sid,l,False)
	ls = freeling_list[3].analyze(ls)
	ls = freeling_list[4].analyze(ls)
	ls = freeling_list[5].analyze(ls)
	ls = freeling_list[6].analyze(ls)
	ls = freeling_list[7].analyze(ls)
	for s in ls:
		ws = s.get_words()
		for w in ws:
			freeling_sentence.append((w.get_form(),w.get_tag().lower()[0]))
	indice_f = 0	
	for i in range(len(sentence_in)):
			if sentence_in[i][2] == 1:
				if sentence_in[i][0] != freeling_sentence[indice_f][0]:
					if sentence_in[i][0].encode("utf-8") not in lista_separables and not palabra_citada(sentence_in[i][0]) and sentence_in[i][0].encode("utf-8") not in ancora and sentence_in[i][0].encode("utf-8") not in free and not es_palabra_con_coma(sentence_in[i][0].encode("utf-8"), freeling_sentence[indice_f][0].encode("utf-8"), freeling_sentence[indice_f+1][0].encode("utf-8")) and not es_palabra_compuesta(sentence_in[i][0], freeling_sentence[indice_f][0], freeling_sentence[indice_f+1][0], sentence_in[i][1], freeling_sentence[indice_f][1]):
						print sentence_in
						print freeling_sentence
						print sentence_in[i][0] + " " + sentence_in[i][1]
						print freeling_sentence[indice_f][0] + " " + freeling_sentence[indice_f][1]
						print freeling_sentence[indice_f + 1][0] + " " + freeling_sentence[indice_f + 1][1]
						raise ValueError('A very specific bad thing happened')
					elif palabra_citada(sentence_in[i][0]):
						i = 0
						while i < 2:
							salida.append(freeling_sentence[indice_f])
							if freeling_sentence[indice_f][0] == "\"":
								i += 1
							indice_f += 1
					elif sentence_in[i][0].encode("utf-8") in lista_separables:
						for k in range(indice_f,indice_f + lista_separables[sentence_in[i][0].encode("utf-8")]):
							salida.append(freeling_sentence[k])
						indice_f += lista_separables[sentence_in[i][0].encode("utf-8")]
					elif es_palabra_con_coma(sentence_in[i][0].encode("utf-8"), freeling_sentence[indice_f][0].encode("utf-8"), freeling_sentence[indice_f+1][0].encode("utf-8")):
						salida.append((freeling_sentence[indice_f]))
						salida.append((freeling_sentence[indice_f+1]))
						indice_f += 2
					elif es_palabra_compuesta(sentence_in[i][0], freeling_sentence[indice_f][0], freeling_sentence[indice_f+1][0], sentence_in[i][1], freeling_sentence[indice_f][1]):
						salida.append((sentence_in[i][0],sentence_in[i][1]))
						indice_f += 2
					elif sentence_in[i][0].encode("utf-8") in free:
						salida.append((sentence_in[i][0],freeling_sentence[indice_f][1]))
						indice_f += free[sentence_in[i][0].encode("utf-8")]
					else:
						salida.append((sentence_in[i][0],sentence_in[i][1]))
						indice_f += ancora[sentence_in[i][0].encode("utf-8")]
				else:
					salida.append(freeling_sentence[indice_f])
					indice_f += 1
			else:
				salida.append((sentence_in[i][0],sentence_in[i][1]))
				#print len(freeling_sentence)
				#print indice_f
				if sentence_in[i][0].encode("utf-8") in lista_separables:
					indice_f += lista_separables[sentence_in[i][0]]
				elif palabra_citada(sentence_in[i][0]):
					i = 0
					while i < 2:
						salida.append(freeling_sentence[indice_f])
						if freeling_sentence[indice_f][0] == "\"":
							i += 1
						indice_f += 1
				elif sentence_in[i][0].encode("utf-8") in ancora:
					indice_f += ancora[sentence_in[i][0].encode("utf-8")]
				elif sentence_in[i][0] != freeling_sentence[indice_f][0]:
					palabra_sin_tilde = quitar_tildes(sentence_in[i][0])
					try:
						palabra_compuesta = freeling_sentence[indice_f][0]
						indice_f += 1
						while palabra_sin_tilde != palabra_compuesta and sentence_in[i][0] != palabra_compuesta:
							palabra_compuesta += freeling_sentence[indice_f][0]
							indice_f += 1
					except:
						print sentence_in[i][0].encode("utf-8")
						print sentence_in[i]
						print freeling_sentence
						raise ValueError('Re que mori igual')
				else:
					indice_f += 1
#		else:
#			if sentence_in[i][0] != freeling_sentence[indice_f][0]:
#
#				print sentence_in[i][0] + " " + sentence_in[i][1]
#				print freeling_sentence[indice_f][0] + " " + freeling_sentence[indice_f][1]
#				print freeling_sentence[indice_f + 1][0] + " " + freeling_sentence[indice_f + 1][1]
#				raise ValueError('A very specific bad thing happened')
#			if sentence_in[i][2] == 1:
#				salida.append(freeling_sentence[indice_f])
#			else:
#				salida.append((sentence_in[i][0],sentence_in[i][1]))
#			indice_f += 1
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
					aux_word = correct_escape_sequences(w)
					sentence.append((aux_word, word[1], 1))
			freeling = True
		elif "_" in word[0]:
			words = word[0].split("_")
			for w in words:
				if w != "":
					aux_word = correct_escape_sequences(w)
					sentence.append((aux_word, word[1], 1))
			freeling = True
		else:
			aux_word = correct_escape_sequences(word[0])
			sentence.append((aux_word, word[1], 0))
	if freeling:
		sentence = process_freeling(sentence, freeling_list)
	for word in sentence:
		aux_uno = number_filter(word[0])
		aux_dos = date_filter(aux_uno)
		pos_tag = vector_variante(opciones_pos[word[1]],largo_vector)
		intermediate.append((aux_dos,pos_tag))
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
				output_file.write(o.encode("utf-8"))
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

mf.set_active_options(False, False, True, False,
                      True, True, False, True,
                      False, True, False, True )

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