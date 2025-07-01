import nltk
from nltk import CFG

# 146_Gramatica_Cusal_Definida.py

# Importamos la librería nltk para trabajar con gramáticas libres de contexto

# Definimos una gramática causal definida simple
# Esta gramática modela oraciones del tipo: "Si X entonces Y"
causal_grammar = CFG.fromstring("""
S -> COND CAUSE
COND -> 'si' EVENTO
CAUSE -> 'entonces' EVENTO
EVENTO -> SUJETO VERBO OBJETO
SUJETO -> 'Juan' | 'Maria'
VERBO -> 'come' | 'estudia'
OBJETO -> 'manzanas' | 'matematicas'
""")

# Creamos un parser usando la gramática definida
parser = nltk.ChartParser(causal_grammar)

# Definimos una oración de ejemplo para analizar
sentence = "si Juan come manzanas entonces Maria estudia matematicas".split()

# Analizamos la oración y mostramos los árboles sintácticos generados
print("Árboles sintácticos posibles para la oración:")
for tree in parser.parse(sentence):
    print(tree)
    tree.pretty_print()
