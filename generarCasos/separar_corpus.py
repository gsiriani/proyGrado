import sys
import os
import re
import random

folder = sys.argv[1]
output_training_file = open("ancora_training.xml","w")
output_testing_file = open("ancora_testing.xml","w")
output_pruebas_file = open("ancora_pruebas.xml","w")

for file in os.listdir(folder):
	open_file = open(folder + "/" + file, "r")
	in_sentence = False
	sentence = []
	n_random = 0
	for o_line in open_file:
		line = o_line.lower()
		if not in_sentence and "<sentence" in line:
			in_sentence = True
			n_random = random.random()
		if in_sentence:
			if n_random <= 0.7:
				output_training_file.write(line)
			elif n_random <= 0.85:
				output_testing_file.write(line)
			else:
				output_pruebas_file.write(line)
		if in_sentence and "</sentence" in line:
			in_sentence = False