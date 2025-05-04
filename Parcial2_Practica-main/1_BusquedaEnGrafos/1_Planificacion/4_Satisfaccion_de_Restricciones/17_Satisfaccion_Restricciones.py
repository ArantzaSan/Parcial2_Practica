from collections import defaultdict

# Clase para representar un grafo con nodos
class NodoGrafo:
    def __init__(self, num_nodos):
        self.num_nodos = num_nodos
        self.conexiones = defaultdict(list)

    def agregar_conexion(self, nodo1, nodo2):
        self.conexiones[nodo1].append(nodo2)
        self.conexiones[nodo2].append(nodo1)

# Verifica si asignar un color a un nodo es válido
def es_valido(grafo, nodo, color_asignado, asignaciones_color):
    for vecino in grafo.conexiones[nodo]:
        if asignaciones_color[vecino] == color_asignado:
            return False
    return True

# Función recursiva para colorear el grafo
def colorear_recursivo(grafo, num_colores, nodo_actual=0, asignaciones_color=None):
    if asignaciones_color is None:
        asignaciones_color = [0] * grafo.num_nodos

    if nodo_actual == grafo.num_nodos:
        return True

    for color in range(1, num_colores + 1):
        if es_valido(grafo, nodo_actual, color, asignaciones_color):
            asignaciones_color[nodo_actual] = color
            if colorear_recursivo(grafo, num_colores, nodo_actual + 1, asignaciones_color):
                return True
            asignaciones_color[nodo_actual] = 0  # Backtrack

    return False

def resolver_problema_coloreo(grafo, num_colores):
    asignaciones_finales = [0] * grafo.num_nodos
    if colorear_recursivo(grafo, num_colores, asignaciones_color=asignaciones_finales):
        return asignaciones_finales
    else:
        return None

# Ejemplo de grafo
mi_grafo = NodoGrafo(4)
mi_grafo.agregar_conexion(0, 1)
mi_grafo.agregar_conexion(0, 2)
mi_grafo.agregar_conexion(1, 2)
mi_grafo.agregar_conexion(1, 3)

# Número de colores disponibles
num_colores = 3

# Resolver el problema de coloreado de grafos
solucion_encontrada = resolver_problema_coloreo(mi_grafo, num_colores)

if solucion_encontrada:
    print(f"Solución encontrada: {solucion_encontrada}")
else:
    print("No se encontró solución.")
