from typing import List, Dict, Any, Tuple

# FOIL (First Order Inductive Learner) para Aprendizaje Basado en Conocimiento
# Ejemplo orientado a una manufacturera para aprender reglas sobre productos defectuosos

# Importamos librerías necesarias

# Definimos una estructura para los ejemplos
class Ejemplo:
    def __init__(self, atributos: Dict[str, Any], es_positivo: bool):
        self.atributos = atributos
        self.es_positivo = es_positivo

# Función para calcular la ganancia de información de un literal
def ganancia_foil(positivos: List[Ejemplo], negativos: List[Ejemplo], literal: Tuple[str, Any]) -> float:
    # Contamos positivos y negativos cubiertos por el literal
    p = sum(1 for e in positivos if e.atributos[literal[0]] == literal[1])
    n = sum(1 for e in negativos if e.atributos[literal[0]] == literal[1])
    if p + n == 0 or p == 0:
        return 0
    total = len(positivos) + len(negativos)
    return p * ( ( (p/(p+n)) if (p+n)>0 else 0 ) - (len(positivos)/total) )

# Función para obtener todos los posibles literales (atributo=valor)
def obtener_literales_posibles(ejemplos: List[Ejemplo], usados: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
    literales = []
    atributos = ejemplos[0].atributos.keys()
    for atributo in atributos:
        valores = set(e.atributos[atributo] for e in ejemplos)
        for valor in valores:
            if (atributo, valor) not in usados:
                literales.append((atributo, valor))
    return literales

# Algoritmo FOIL principal
def foil(ejemplos: List[Ejemplo]) -> List[List[Tuple[str, Any]]]:
    reglas = []
    positivos = [e for e in ejemplos if e.es_positivo]
    negativos = [e for e in ejemplos if not e.es_positivo]

    # Mientras haya ejemplos positivos sin cubrir
    while positivos:
        regla = []
        usados = []
        cubiertos = positivos.copy()
        no_cubiertos = negativos.copy()

        # Construimos una regla añadiendo literales
        while no_cubiertos:
            literales = obtener_literales_posibles(cubiertos + no_cubiertos, usados)
            # Seleccionamos el literal con mayor ganancia
            mejor_literal = max(literales, key=lambda l: ganancia_foil(cubiertos, no_cubiertos, l))
            regla.append(mejor_literal)
            usados.append(mejor_literal)
            # Filtramos ejemplos cubiertos por el literal
            cubiertos = [e for e in cubiertos if e.atributos[mejor_literal[0]] == mejor_literal[1]]
            no_cubiertos = [e for e in no_cubiertos if e.atributos[mejor_literal[0]] == mejor_literal[1]]
            if not cubiertos:
                break
        reglas.append(regla)
        # Eliminamos los positivos cubiertos por la nueva regla
        positivos = [e for e in positivos if not all(e.atributos[a] == v for a, v in regla)]
    return reglas

# Ejemplo de uso en una manufacturera
if __name__ == "__main__":
    # Creamos ejemplos: cada uno representa un producto con atributos y si es defectuoso
    ejemplos = [
        Ejemplo({'color': 'rojo', 'peso': 'ligero', 'tamaño': 'grande'}, True),
        Ejemplo({'color': 'azul', 'peso': 'pesado', 'tamaño': 'pequeño'}, False),
        Ejemplo({'color': 'rojo', 'peso': 'pesado', 'tamaño': 'grande'}, True),
        Ejemplo({'color': 'azul', 'peso': 'ligero', 'tamaño': 'grande'}, False),
        Ejemplo({'color': 'rojo', 'peso': 'ligero', 'tamaño': 'pequeño'}, False),
        Ejemplo({'color': 'rojo', 'peso': 'pesado', 'tamaño': 'pequeño'}, True),
    ]

    # Ejecutamos FOIL para aprender reglas que predicen productos defectuosos
    reglas = foil(ejemplos)

    # Mostramos las reglas aprendidas
    print("Reglas aprendidas para identificar productos defectuosos:")
    for i, regla in enumerate(reglas):
        condiciones = " Y ".join([f"{a}={v}" for a, v in regla])
        print(f"Regla {i+1}: SI {condiciones} ENTONCES Defectuoso")
