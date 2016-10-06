#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
from codecs import open, BOM_UTF8
import re

# Cargo lexicon wikipedia
corpus_wiki = {}
total_ocurrencias_wiki = 0
for linea in open('./corpus_wikipedia/lexicon_wiki-total.txt', encoding="utf-8"):
	r = linea.split()
	if len(r) > 1:
		num = int(re.sub("[^0-9]", "", r[0]))
		corpus_wiki[r[1]] = num
		total_ocurrencias_wiki += num
	

# Cargo lexicon ancora
corpus_ancora = {}
total_ocurrencias_ancora = 0
for linea in open('./corpus_ancora/lexicon_ancora.txt', encoding="utf-8"):
	r = linea.split()
	num = int(re.sub("[^0-9]", "", r[0]))
	corpus_ancora[r[1]] = num
	total_ocurrencias_ancora += num


# Cargo lexicon final
lexicon = []
for linea in open('./lexicon.txt', encoding="utf-8"):
	r = linea.split()
	lexicon.append(r[0])


ocurrencias_wiki = 0
ocurrencias_ancora = 0

for palabra in lexicon:
	if palabra in corpus_wiki:
		ocurrencias_wiki += corpus_wiki[palabra]
	else:
		print palabra
	if palabra in corpus_ancora:
		ocurrencias_ancora += corpus_ancora[palabra]


print 'Porcentaje de cobertura Wikipedia: ' + str(float(len(lexicon)*100/len(corpus_wiki))) + '%'
print 'Porcentaje de ocurrencias cubiertas en Wikipedia: ' + str(float(ocurrencias_wiki*100/total_ocurrencias_wiki)) + '%'

print 'Porcentaje de cobertura Ancora: ' + str(float(len(lexicon)*100/len(corpus_ancora)))  + '%'
print 'Porcentaje de ocurrencias cubiertas: ' + str(float(ocurrencias_ancora*100/total_ocurrencias_ancora)) + '%'

