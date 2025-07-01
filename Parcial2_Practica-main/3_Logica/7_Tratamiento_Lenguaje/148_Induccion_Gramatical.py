import re
from collections import defaultdict

# 148_Induccion_Gramatical.py
# Implementación básica de inducción gramatical para secuencias de texto
# Autor: GitHub Copilot


# Sección 1: Definición de funciones auxiliares

def tokenize(text):
    """
    Tokeniza el texto en palabras usando expresiones regulares.
    """
    return re.findall(r'\w+', text.lower())

def extract_ngrams(tokens, n):
    """
    Extrae n-gramas de una lista de tokens.
    """
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# Sección 2: Inducción de reglas gramaticales

def induce_grammar(sentences, max_ngram=3, min_freq=2):
    """
    Induce reglas gramaticales simples a partir de una lista de oraciones.
    Busca patrones repetidos (n-gramas) y los convierte en reglas.
    """
    ngram_freq = defaultdict(int)
    for sentence in sentences:
        tokens = tokenize(sentence)
        for n in range(2, max_ngram+1):
            for ngram in extract_ngrams(tokens, n):
                ngram_freq[ngram] += 1

    # Selecciona n-gramas frecuentes como reglas
    rules = {}
    rule_id = 1
    for ngram, freq in ngram_freq.items():
        if freq >= min_freq:
            rule_name = f"R{rule_id}"
            rules[rule_name] = ngram
            rule_id += 1

    return rules

# Sección 3: Aplicación de reglas a nuevas oraciones

def apply_rules(sentence, rules):
    """
    Aplica las reglas inducidas a una oración, reemplazando los patrones por el nombre de la regla.
    """
    tokens = tokenize(sentence)
    output = tokens[:]
    for rule_name, pattern in rules.items():
        pattern_len = len(pattern)
        i = 0
        while i <= len(output) - pattern_len:
            if tuple(output[i:i+pattern_len]) == pattern:
                output[i:i+pattern_len] = [rule_name]
                i += 1
            else:
                i += 1
    return ' '.join(output)

# Sección 4: Ejemplo de uso

if __name__ == "__main__":
    # Oraciones de ejemplo
    corpus = [
        "el perro come carne",
        "el gato come pescado",
        "el perro duerme",
        "el gato duerme",
        "el perro come pescado"
    ]

    # Inducción de reglas gramaticales
    reglas = induce_grammar(corpus, max_ngram=3, min_freq=2)
    print("Reglas inducidas:")
    for nombre, patron in reglas.items():
        print(f"{nombre}: {' '.join(patron)}")

    # Aplicación de reglas a una nueva oración
    nueva_oracion = "el perro come pescado"
    resultado = apply_rules(nueva_oracion, reglas)
    print("\nOración original:", nueva_oracion)
    print("Oración con reglas:", resultado)
