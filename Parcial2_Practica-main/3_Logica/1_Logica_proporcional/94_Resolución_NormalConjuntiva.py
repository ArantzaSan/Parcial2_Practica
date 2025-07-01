from typing import List, Set

# Resolución en Forma Normal Conjuntiva (FNC) para lógica proposicional
# Este programa está orientado para ayudar a policías a resolver problemas lógicos
# como deducciones basadas en hechos y reglas.


def clausula_a_conjunto(clausula: str) -> Set[str]:
    """
    Convierte una cláusula en forma de texto a un conjunto de literales.
    Ejemplo: "A OR NOT B" -> {"A", "NOT B"}
    """
    return set(literal.strip() for literal in clausula.split("OR"))

def resolver(clausulas: List[Set[str]]) -> bool:
    """
    Aplica el método de resolución para determinar si un conjunto de cláusulas en FNC es satisfacible.
    """
    while True:
        nuevas_clausulas = set()
        # Iterar sobre todas las combinaciones de pares de cláusulas
        for i in range(len(clausulas)):
            for j in range(i + 1, len(clausulas)):
                resolvente = resolver_clausulas(clausulas[i], clausulas[j])
                if resolvente is not None:
                    # Si el resolvente es vacío, se ha derivado una contradicción
                    if not resolvente:
                        return True
                    nuevas_clausulas.add(frozenset(resolvente))
        
        # Si no se generan nuevas cláusulas, detener el proceso
        if nuevas_clausulas.issubset(map(frozenset, clausulas)):
            return False
        
        # Agregar las nuevas cláusulas al conjunto de cláusulas
        clausulas.extend(map(set, nuevas_clausulas))

def resolver_clausulas(clausula1: Set[str], clausula2: Set[str]) -> Set[str]:
    """
    Intenta resolver dos cláusulas. Si tienen literales complementarios, devuelve el resolvente.
    """
    for literal in clausula1:
        complemento = "NOT " + literal if not literal.startswith("NOT ") else literal[4:]
        if complemento in clausula2:
            # Crear el resolvente eliminando los literales complementarios
            resolvente = (clausula1 | clausula2) - {literal, complemento}
            return resolvente
    return None

def main():
    """
    Función principal del programa.
    Aquí se definen los hechos y reglas en forma de cláusulas en FNC.
    """
    print("Bienvenido al sistema de resolución lógica para policías.")
    print("Este programa ayuda a deducir conclusiones basadas en hechos y reglas.")

    # Ejemplo de entrada: conjunto de cláusulas en FNC
    # Cada cláusula es una disyunción de literales (variables o sus negaciones)
    clausulas = [
        clausula_a_conjunto("NOT Sospechoso OR TieneCoartada"),
        clausula_a_conjunto("NOT TieneCoartada OR Culpable"),
        clausula_a_conjunto("Sospechoso"),
        clausula_a_conjunto("NOT Culpable")
    ]

    # Aplicar el método de resolución
    contradiccion = resolver(clausulas)

    if contradiccion:
        print("Se ha encontrado una contradicción. El caso tiene inconsistencias.")
    else:
        print("No se encontraron contradicciones. El caso es consistente.")

if __name__ == "__main__":
    main()
