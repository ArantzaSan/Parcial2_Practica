import nltk
from nltk.translate import IBMModel1
from nltk.corpus import comtrans
from nltk.tokenize import word_tokenize

# Cargar un corpus paralelo pequeño de ejemplo (inglés-español)
corpus_ejemplo = comtrans.aligned_sents('alignment-en-es.txt')[:5]

# Entrenar el Modelo IBM Modelo 1
modelo_ibm1 = IBMModel1(corpus_ejemplo, iterations=5)

# Función para traducir una frase usando el modelo
def traducir_estadisticamente(frase_ingles, modelo):
    tokens_ingles = word_tokenize(frase_ingles)
    traduccion = []
    for palabra_ingles in tokens_ingles:
        mejor_probabilidad = 0.0
        mejor_traduccion = None
        for palabra_espanol in modelo.translation_table[palabra_ingles]:
            probabilidad = modelo.translation_table[palabra_ingles][palabra_espanol]
            if probabilidad > mejor_probabilidad:
                mejor_probabilidad = probabilidad
                mejor_traduccion = palabra_espanol
        if mejor_traduccion:
            traduccion.append(mejor_traduccion)
        else:
            traduccion.append(palabra_ingles) # Si no se encuentra traducción, se mantiene la palabra original
    return " ".join(traduccion)

# Frase en inglés para traducir
frase_a_traducir = "the cat sat on the mat"

# Traducir la frase
traduccion_espanol = traducir_estadisticamente(frase_a_traducir, modelo_ibm1)

print(f"Frase en inglés: {frase_a_traducir}")
print(f"Traducción (IBM Modelo 1): {traduccion_espanol}")
