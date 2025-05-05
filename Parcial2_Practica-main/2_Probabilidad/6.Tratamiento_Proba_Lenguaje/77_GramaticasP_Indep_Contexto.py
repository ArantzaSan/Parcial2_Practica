import nltk

def verificar_gic(gramatica, cadena):
    """Verifica si una cadena es generada por una Gramática Independiente de Contexto (GIC)."""
    parser = nltk.ChartParser(gramatica)
    for tree in parser.parse(cadena.split()):
        print("Cadena aceptada por la GIC:")
        tree.pretty_print()
        return True
    print("Cadena rechazada por la GIC.")
    return False

# Ejemplo de GIC
gramatica_ejemplo = nltk.CFG.fromstring("""
    S -> NP VP
    NP -> Det N | N
    VP -> V | V NP
    Det -> 'el' | 'la'
    N -> 'gato' | 'perro' | 'pelota'
    V -> 'come' | 'persigue'
""")

# Cadenas de prueba
cadena_aceptada = "el gato come"
cadena_rechazada = "gato el come"

print(f"Verificando '{cadena_aceptada}':")
verificar_gic(gramatica_ejemplo, cadena_aceptada)

print(f"\nVerificando '{cadena_rechazada}':")
verificar_gic(gramatica_ejemplo, cadena_rechazada)
