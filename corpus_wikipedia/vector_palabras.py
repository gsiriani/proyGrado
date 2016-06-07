from codecs import open, BOM_UTF8

def is_number(token):
	try:
		n = float(token)
		return "NUM"
	except:
		return token

class palabras_comunes:

	def __init__(self, archivo):

		i = 0
		self.dic_p = {}
		for p in open(archivo, encoding="latin-1"):
			self.dic_p[p.split()[0]] = i
			i = i + 1
		self.dic_p["NUM"] = i
		self.UNK = i + 1

	def obtener_indice(self,token):
		p = token.lower()
		c = is_number(p)
		return self.dic_p.setdefault(c, self.UNK)