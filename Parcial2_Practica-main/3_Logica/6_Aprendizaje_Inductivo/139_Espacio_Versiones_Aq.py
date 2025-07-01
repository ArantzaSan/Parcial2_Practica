from typing import List, Tuple, Any

"""
Espacio de Versiones AQ - Implementación en Python
--------------------------------------------------
Este programa implementa el algoritmo de espacio de versiones AQ para aprendizaje basado en conocimiento.
La arquitectura está dividida en módulos para facilitar la comprensión y mantenimiento.
"""

# ------------------------------
# 1. Definición de estructuras
# ------------------------------


# Un ejemplo es una tupla (atributos, clase)
Example = Tuple[List[Any], str]
# Una regla es una lista de condiciones (pueden ser valores o '?')
Rule = List[Any]

# ------------------------------
# 2. Funciones de utilidad
# ------------------------------

def cubre(regla: Rule, ejemplo: Example) -> bool:
    """Verifica si una regla cubre un ejemplo."""
    atributos, _ = ejemplo
    return all(r == '?' or r == a for r, a in zip(regla, atributos))

def generaliza(regla: Rule, ejemplo: Example) -> Rule:
    """Generaliza una regla para cubrir un ejemplo positivo."""
    return ['?' if r != a else r for r, a in zip(regla, ejemplo[0])]

def es_mas_general(r1: Rule, r2: Rule) -> bool:
    """Verifica si r1 es más general que r2."""
    mas_general = False
    for a, b in zip(r1, r2):
        if a == '?' and b != '?':
            mas_general = True
        elif a != '?' and b == '?':
            return False
        elif a != b:
            return False
    return mas_general

# ------------------------------
# 3. Algoritmo AQ principal
# ------------------------------

def aq(positivos: List[Example], negativos: List[Example]) -> List[Rule]:
    """
    Implementa el algoritmo AQ para encontrar reglas que cubran todos los positivos
    y excluyan los negativos.
    """
    reglas = []
    for p in positivos:
        # Inicialmente, la regla es el ejemplo positivo
        regla = list(p[0])
        # Generaliza la regla para excluir negativos
        for n in negativos:
            if cubre(regla, n):
                regla = generaliza(regla, n)
        # Elimina reglas subsumidas
        reglas = [r for r in reglas if not es_mas_general(regla, r)]
        if not any(es_mas_general(r, regla) for r in reglas):
            reglas.append(regla)
    return reglas

# ------------------------------
# 4. Ejemplo de uso
# ------------------------------

if __name__ == "__main__":
    # Definición de ejemplos (atributos, clase)
    ejemplos = [
        (['rojo', 'grande', 'redondo'], 'positivo'),
        (['azul', 'grande', 'redondo'], 'negativo'),
        (['rojo', 'pequeño', 'redondo'], 'positivo'),
        (['rojo', 'grande', 'cuadrado'], 'negativo'),
    ]

    positivos = [e for e in ejemplos if e[1] == 'positivo']
    negativos = [e for e in ejemplos if e[1] == 'negativo']

    # Ejecuta el algoritmo AQ
    reglas = aq(positivos, negativos)

    # Muestra las reglas encontradas
    print("Reglas generadas por AQ:")
    for i, r in enumerate(reglas, 1):
        print(f"Regla {i}: {r}")

# ------------------------------
# Fin del programa
# ------------------------------
