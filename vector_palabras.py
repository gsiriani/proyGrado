from codecs import open, BOM_UTF8

def is_punct(token):
	# Retorna True si el token es un caracter no alfanumerico de un digito
	return ((len(token) == 1) && !(token.isalnum()))


class palabras_comunes:

	def __init__(self, archivo):
		# Genera un diccionario a partir de un archivo de palabras y palabras auxiliares
		i = 0
		self.dic_p = {}
		for p in open(archivo, encoding="latin-1"):
			self.dic_p[p.split()[0]] = i
			i = i + 1
		self.dic_p["NUM"] = i
		self.dic_p["OUT"] = i + 1
		self.dic_p["PUNCT"] = i + 2
		self.UNK = i + 3

	def obtener_indice(self,token):
		# Retorna el indice del token en el diccionario
		if (token == "OUT"):
			return self.dic_p["OUT"]
		if (token == "NUM"):
			return self.dic_p["NUM"]
		if is_punct(token):
			return self.dic_p["PUNCT"]
		p = token.lower()
		return self.dic_p.setdefault(c, self.UNK)