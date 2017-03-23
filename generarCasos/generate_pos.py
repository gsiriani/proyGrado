import sys
import os
import re
import random

# Orden de los tags en aparicion
orden_tags = {"pos" : 0,
			  "gen" : 1,
			  "num" : 2,
			  "posfunction" : 3,
			  "postype" : 4,
			  "possessornum" : 5,
			  "punct" : 6,
			  "punctenclose" : 7,
			  "case" : 8,
			  "person" : 9,
			  "polite" : 10,
			  "mood" : 11,
			  "tense" : 12}

# Cantidad de opciones por tag segun el orden
cantidad_opciones = [12, 3, 3, 1, 25, 3, 16, 2, 4, 3, 2, 2, 5]

# Opciones para cada ta
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

opciones_gen = {"c" : 0,
				"f" : 1,
				"m" : 2}

opciones_num = {"c" : 0,
				"p" : 1,
				"s" : 2}

opciones_posfunction = {"participle" : 0}

opciones_postype = {"qualificative" : 0,
					"ordinal" : 1,
					"article" : 2,
					"demonstrative" : 3,
					"exclamative" : 4,
					"indefinite" : 5,
					"interrogative" : 6,
					"numeral" : 7,
					"ordinal" : 8,
					"possessive" : 9,
					"common" : 10,
					"proper" : 11,
					"personal" : 12,
					"possessive" : 13,
					"relative" : 14,
					"general" : 15,
					"negative" : 16,
					"auxiliary" : 17,
					"semiauxiliary" : 18,
					"currency" : 19,
					"percentage" : 20,
					"preposition" : 21,
					"coordinating" : 22,
					"main" : 23,
					"subordinating" : 24}

opciones_possessornum = {"c" : 0,
						 "p" : 1,
						 "s" : 2}

opciones_punct = {"apostrophe" : 0,
				  "bracket" : 1,
				  "sqbracket" : 2,
				  "cubracket" : 3,
				  "colon" : 4,
				  "comma" : 5,
				  "etc" : 6,
				  "exclamationmark" : 7,
				  "hyphen" : 8,
				  "mathsign" : 9,
				  "period" : 10,
				  "questionmark" : 11,
				  "quotation" : 12,
				  "semicolon" : 13,
				  "slash" : 14,
				  "revslash" : 15}

opciones_punctenclose = {"open" : 0,
						 "close" : 1}

opciones_case = {"accusative" : 0,
				 "dative" : 1,
				 "nominative" : 2,
				 "oblique" : 3}

opciones_person = {"1" : 0,
				   "2" : 1,
				   "3" : 2}

opciones_polite = {"yes" : 0,
				   "no" : 1}

opciones_mood = {"gerund" : 0,
				 "imperative" : 1,
				 "indicative" : 2,
				 "infinitive" : 3,
				 "participle" : 4,
				 "subjunctive" : 5}

opciones_tense = {"conditional" : 0,
				  "future" : 1,
				  "imperfect" : 2,
				  "past" : 3,
				  "present" : 4}

# Diccionario de opciones de tag
opciones_tags = {"pos" : opciones_pos,
				 "gen" : opciones_gen,
				 "num" : opciones_num,
				 "posfunction" : opciones_posfunction,
				 "postype" : opciones_postype,
				 "possessornum" : opciones_possessornum,
				 "punct" : opciones_punct,
				 "punctenclose" : opciones_punctenclose,
				 "case" : opciones_case,
				 "person" : opciones_person,
				 "polite" : opciones_polite,
				 "mood" : opciones_mood,
				 "tense" : opciones_tense}

def sumatoria(arreglo):
	total = 0
	for i in arreglo:
		total += i
	return total

def put_one(output, tag, opcion):
	posicion = sumatoria(cantidad_opciones[:orden_tags[tag]]) + opciones_tags[tag][opcion]
	output[posicion] = 1
	return output

def array_to_str(arreglo):
	string = ""
	for p in arreglo:
		if string == "":
			string = str(p)
		else:
			string += " " + str(p)
	return string

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
		if words[i][2] == 0:
			line = ""
			max_index = min(i + 6,len(words))
			min_index = max(0,i - 5)
			for j in range(0,5 - i):
				line += "OUT "
			for j in range(min_index, max_index):
				line += words[j][0] + " "
			for j in range(6 - (len(words) - i)):
				line += "OUT "
			line += array_to_str(words[i][1]) + "\n"
			output.append((line,0))
	return output

def generate_freeling_cases(sentence):
	oracion = "<sentence>"
	words = "\n"
	for word in sentence:
		oracion += " " + word[0]
		if word[2] == 1:
			words += "<word>" + word[0] + "\n"
	oracion += words
	return (oracion, 1)

def get_postag(line):
	postag = []
	for i in range(sumatoria(cantidad_opciones)):
		postag.append(0)
	pos = re.sub(".*?<","",line)[0]
	postag = put_one(postag, "pos", pos)
	if " gen=" in line:
		gen = re.sub(".*? gen=\"","",line)
		gen = re.sub("\".*\n", "", gen)
		postag = put_one(postag, "gen", gen)
	if " num=" in line:
		num = re.sub(".*? num=\"","",line)
		num = re.sub("\".*\n", "", num)
		postag = put_one(postag, "num", num)
	if " posfunction=" in line:
		posfunction = re.sub(".*? posfunction=\"","",line)
		posfunction = re.sub("\".*\n", "", posfunction)
		postag = put_one(postag, "posfunction", posfunction)
	if " postype=" in line:
		postype = re.sub(".*? postype=\"","",line)
		postype = re.sub("\".*\n", "", postype)
		postag = put_one(postag, "postype", postype)
	if " possessornum=" in line:
		possessornum = re.sub(".*? possessornum=\"","",line)
		possessornum = re.sub("\".*\n", "", possessornum)
		postag = put_one(postag, "possessornum", possessornum)
	if " punct=" in line:
		punct = re.sub(".*? punct=\"","",line)
		punct = re.sub("\".*\n", "", punct)
		postag = put_one(postag, "punct", punct)
	if " punctenclose=" in line:
		punctenclose = re.sub(".*? punctenclose=\"","",line)
		punctenclose = re.sub("\".*\n", "", punctenclose)
		postag = put_one(postag, "punctenclose", punctenclose)
	if " case=" in line:
		case = re.sub(".*? case=\"","",line)
		case = re.sub("\".*\n", "", case)
		postag = put_one(postag, "case", case)
	if " person=" in line:
		person = re.sub(".*? person=\"","",line)
		person = re.sub("\".*\n", "", person)
		postag = put_one(postag, "person", person)
	if " polite=" in line:
		polite = re.sub(".*? polite=\"","",line)
		polite = re.sub("\".*\n", "", polite)
		postag = put_one(postag, "polite", polite)
	if " mood=" in line:
		mood = re.sub(".*? mood=\"","",line)
		mood = re.sub("\".*\n", "", mood)
		postag = put_one(postag, "mood", mood)
	if " tense=" in line:
		tense = re.sub(".*? tense=\"","",line)
		tense = re.sub("\".*\n", "", tense)
		postag = put_one(postag, "tense", tense)
	return postag

def process_sentence(sentence_in):
	intermediate = []
	sentence = []
	freeling = False
	for word in sentence_in:
		if "_" in word[0]:
			words = word[0].split("_")
			for w in words:
				aux_uno = number_filter(w)
				aux_dos = date_filter(aux_uno)
				aux_tres = correct_escape_sequences(aux_dos)
				sentence.append((aux_tres,word[1],1))
				freeling = True
		else:
			aux_uno = number_filter(word[0])
			aux_dos = date_filter(aux_uno)
			aux_tres = correct_escape_sequences(aux_dos)
			sentence.append((aux_tres,word[1],0))
	for word in sentence:
		postag = get_postag(word[1])
		intermediate.append((word[0],postag,word[2]))
	output = generate_cases(intermediate)
	if freeling:
		# En realidad aca iria el procesamiento con freeling de la frase pero aun
		# no lo tengo esto implementado. Por ahora solo lo separo en otro archivo.
		output.append(generate_freeling_cases(sentence))
	return output

folder = sys.argv[1]
output_training_file = open("pos_training.csv","w")
output_testing_file = open("pos_testing.csv","w")
output_pruebas_file = open("pos_pruebas.csv","w")
output_file_freeling = open("freeling_pos.csv","w")

for file in os.listdir(folder):
	open_file = open(folder + "/" + file, "r")
	in_sentence = False
	sentence = []
	for o_line in open_file:
		line = o_line.lower()
		if not in_sentence and "<sentence" in line:
			in_sentence = True
		if in_sentence and " wd=" in line:
			aux_line = re.sub(".*?wd=\"","",line)
			word = re.sub("\".*\n","",aux_line)
			sentence.append((word, line))
		if in_sentence and "</sentence" in line:
			in_sentence = False
			output = process_sentence(sentence)
			r = random.random()
			for o in output:
				if o[1] == 0:
					if r <= 0.7:
						output_training_file.write(o[0])
					elif r <= 0.85:
						output_pruebas_file.write(o[0])
					else:
						output_testing_file.write(o[0])
				else:
					output_file_freeling.write(o[0])
			sentence = []