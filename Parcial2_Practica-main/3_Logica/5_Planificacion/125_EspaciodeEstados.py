from collections import deque

# Espacio de Estados como recurso de planificación
# Autor: GitHub Copilot

# Definimos una clase para representar un estado
class Estado:
    def __init__(self, nombre, datos=None):
        self.nombre = nombre
        self.datos = datos or {}

    def __eq__(self, other):
        return isinstance(other, Estado) and self.nombre == other.nombre and self.datos == other.datos

    def __hash__(self):
        return hash((self.nombre, frozenset(self.datos.items())))

    def __repr__(self):
        return f"Estado({self.nombre}, {self.datos})"

# Definimos una clase para representar una acción
class Accion:
    def __init__(self, nombre, precondicion, efecto):
        self.nombre = nombre
        self.precondicion = precondicion  # función: Estado -> bool
        self.efecto = efecto              # función: Estado -> Estado

    def es_aplicable(self, estado):
        return self.precondicion(estado)

    def aplicar(self, estado):
        return self.efecto(estado)

    def __repr__(self):
        return f"Accion({self.nombre})"

# Definimos el espacio de estados y el algoritmo de planificación (búsqueda en anchura)
def planificar(estado_inicial, estado_objetivo, acciones):

    frontera = deque()
    frontera.append((estado_inicial, []))
    visitados = set()

    while frontera:
        estado_actual, plan = frontera.popleft()
        if estado_actual == estado_objetivo:
            return plan  # Devuelve la secuencia de acciones

        visitados.add(estado_actual)

        for accion in acciones:
            if accion.es_aplicable(estado_actual):
                nuevo_estado = accion.aplicar(estado_actual)
                if nuevo_estado not in visitados:
                    frontera.append((nuevo_estado, plan + [accion.nombre]))

    return None  # No se encontró plan

# Ejemplo de uso:
if __name__ == "__main__":
    # Definimos estados
    estado_inicial = Estado("inicio", {"pos": "A"})
    estado_objetivo = Estado("fin", {"pos": "C"})

    # Definimos acciones
    acciones = [
        Accion(
            "ir_A_B",
            lambda e: e.datos["pos"] == "A",
            lambda e: Estado("intermedio", {"pos": "B"})
        ),
        Accion(
            "ir_B_C",
            lambda e: e.datos["pos"] == "B",
            lambda e: Estado("fin", {"pos": "C"})
        ),
    ]

    # Ejecutamos el planificador
    plan = planificar(estado_inicial, estado_objetivo, acciones)

    # Mostramos el resultado
    if plan:
        print("Plan encontrado:", plan)
    else:
        print("No se encontró un plan.")
