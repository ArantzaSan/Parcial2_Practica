import nltk
from nltk import CFG

# 144_Analisis_Sintactico.py

# Importamos la librería necesaria para definir la gramática

# Definimos una gramática libre de contexto simple para oraciones en español
# S: oración, NP: sintagma nominal, VP: sintagma verbal, Det: determinante, N: nombre, V: verbo
grammar = CFG.fromstring("""
    S -> NP VP
    NP -> Det N
    VP -> V NP | V
    Det -> 'el' | 'la'
    N -> 'gato' | 'perro' | 'niña' | 'niño'
    V -> 'come' | 've' | 'juega'
""")

# Creamos un analizador sintáctico descendente
parser = nltk.ChartParser(grammar)

# Definimos una oración de ejemplo para analizar
sentence = "el gato come".split()

# Analizamos la oración y mostramos los árboles sintácticos posibles
print("Árbol(es) sintáctico(s) para la oración:", ' '.join(sentence))
for tree in parser.parse(sentence):
    print(tree)
    tree.pretty_print()  # Muestra el árbol de manera visual

# Nota:
# - Puedes modificar la gramática y las oraciones para experimentar con diferentes estructuras.
# - Asegúrate de tener instalado NLTK: pip install nltk
