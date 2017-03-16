from codecs import open, BOM_UTF8

class palabras_comunes:

	def __init__(self, archivo):
		# Genera un diccionario a partir de un archivo de palabras y palabras auxiliares
		i = 0
		self.dic_p = {}
		for p in open(archivo):
			self.dic_p[p.split()[0]] = i
			i = i + 1
		self.dic_p["PUNCT"] = i
		self.dic_p["NUM"] = i + 1
		self.dic_p["OUT"] = i + 2
		self.UNK = i + 3

	def obtener_indice(self,token):
		# Retorna el indice del token en el diccionario
		if (token == "OUT"):
			return self.dic_p["OUT"]
		if (token == "NUM"):
			return self.dic_p["NUM"]
		if self.is_punct(token):
			return self.dic_p["PUNCT"]
		p = token.lower()
		return self.dic_p.setdefault(p, self.UNK)

	def is_punct(self, token):
		# Retorna True si el token es un caracter no alfanumerico de un digito que no esta en el diccionario
		return ((not (token in self.dic_p.keys())) and (len(token) == 1) and not (token.isalnum()))