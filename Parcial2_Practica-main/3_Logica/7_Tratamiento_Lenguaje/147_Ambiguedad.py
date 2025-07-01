import nltk
from nltk import CFG
from nltk.parse.chart import ChartParser

# 147_Ambiguedad.py
# Ejemplo de tratamiento de ambigüedad en lenguaje natural usando Python y NLTK


# Descarga los recursos necesarios de NLTK
nltk.download('punkt')

# Definición de una gramática ambigua
# La frase "Veo al hombre con el telescopio" puede interpretarse de dos maneras:
# 1. Yo uso el telescopio para ver al hombre.
# 2. El hombre tiene el telescopio y yo lo veo.
grammar = CFG.fromstring("""
    S -> NP VP
    NP -> Det N | Det N PP
    VP -> V NP | VP PP
    PP -> P NP
    Det -> 'el' | 'al'
    N -> 'hombre' | 'telescopio'
    V -> 'veo'
    P -> 'con'
""")

# Frase ambigua a analizar
sentence = "veo al hombre con el telescopio"

# Tokenización de la frase
tokens = nltk.word_tokenize(sentence)

# Creación del parser con la gramática ambigua
parser = ChartParser(grammar)

# Parseo de la frase y muestra de los árboles sintácticos posibles
print("Árboles sintácticos posibles para la frase ambigua:")
for tree in parser.parse(tokens):
    print(tree)
    tree.pretty_print()
