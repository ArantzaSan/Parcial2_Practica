import nltk

def verificar_gl(gramatica_lexicalizada_str, cadena):
    """Verifica si una cadena es reconocida por una Gramática Lexicalizada (GL)."""
    gramatica = nltk.DependencyGrammar.fromstring(gramatica_lexicalizada_str)
    parser = nltk.ProjectiveDependencyParser(gramatica)
    resultado = list(parser.parse(cadena.split()))
    if resultado:
        print("Cadena reconocida por la Gramática Lexicalizada:")
        for arbol in resultado:
            print(arbol.tree())
        return True
    else:
        print("Cadena no reconocida por la Gramática Lexicalizada.")
        return False

# Ejemplo de Gramática Lexicalizada (basada en dependencias)
gramatica_lexicalizada_ejemplo = """
    'come' -> 'gato' | 'pescado'
    'gato' -> 'el'
    'pescado' -> 'un'
"""

# Cadenas de prueba
cadena_reconocida = "el gato come un pescado"
cadena_no_reconocida = "gato come el pescado un"

print(f"Verificando '{cadena_reconocida}':")
verificar_gl(gramatica_lexicalizada_ejemplo, cadena_reconocida)

print(f"\nVerificando '{cadena_no_reconocida}':")
verificar_gl(gramatica_lexicalizada_ejemplo, cadena_no_reconocida)
