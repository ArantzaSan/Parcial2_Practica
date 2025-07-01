"""
140_Explicaciones_Info_Relevante.py

Este programa simula un buscador web sencillo que, además de mostrar resultados,
proporciona explicaciones e información relevante sobre los términos buscados.
Cada sección del código está comentada para facilitar su comprensión y extensión.
"""

# 1. Base de datos simulada: términos y explicaciones relevantes
knowledge_base = {
    "python": {
        "descripcion": "Python es un lenguaje de programación interpretado, de alto nivel y propósito general.",
        "info_relevante": [
            "Python es ampliamente usado en ciencia de datos, inteligencia artificial y desarrollo web.",
            "Su sintaxis simple facilita el aprendizaje para principiantes."
        ]
    },
    "inteligencia artificial": {
        "descripcion": "La inteligencia artificial es la simulación de procesos de inteligencia humana por máquinas.",
        "info_relevante": [
            "Incluye aprendizaje automático, procesamiento de lenguaje natural y visión por computadora.",
            "Se utiliza en asistentes virtuales, autos autónomos y sistemas de recomendación."
        ]
    },
    "buscador web": {
        "descripcion": "Un buscador web es una herramienta que permite encontrar información en Internet.",
        "info_relevante": [
            "Ejemplos populares: Google, Bing, DuckDuckGo.",
            "Utilizan algoritmos para indexar y clasificar páginas web."
        ]
    }
}

# 2. Función para buscar un término y mostrar explicaciones e información relevante
def buscar_termino(termino):
    termino = termino.lower()
    if termino in knowledge_base:
        print(f"\nTérmino encontrado: {termino.capitalize()}")
        print("Descripción:")
        print(f"  {knowledge_base[termino]['descripcion']}")
        print("Información relevante:")
        for info in knowledge_base[termino]['info_relevante']:
            print(f"  - {info}")
    else:
        print(f"\nNo se encontró información sobre '{termino}'. Intenta con otro término.")

# 3. Interfaz principal: permite al usuario buscar términos hasta que decida salir
def main():
    print("=== Buscador Web con Explicaciones e Información Relevante ===")
    print("Escribe un término para buscar (o 'salir' para terminar):")
    while True:
        termino = input("> ")
        if termino.lower() == "salir":
            print("¡Hasta luego!")
            break
        buscar_termino(termino)

# 4. Punto de entrada del programa
if __name__ == "__main__":
    main()
