import heapq

def encontrar_camino_minimo(red_vial, inicio, destino):
    """
    Implementa la búsqueda de costo uniforme para encontrar el camino de menor costo
    desde un nodo de inicio hasta un nodo de destino en un grafo ponderado.

    Args:
        red_vial (dict): Un diccionario que representa el grafo. Las claves son los nodos
                         y los valores son listas de tuplas (vecino, costo).
        inicio: El nodo de inicio de la búsqueda.
        destino: El nodo objetivo de la búsqueda.

    Returns:
        float: El costo mínimo para llegar al nodo destino desde el nodo inicio.
               Retorna infinito si no se encuentra un camino.
    """
    frontera = [(0, inicio)]  # Cola de prioridad (costo, nodo)
    costos_acumulados = {inicio: 0}
    nodos_explorados = set()

    while frontera:
        costo_actual, nodo_actual = heapq.heappop(frontera)

        if nodo_actual == destino:
            return costo_actual

        if nodo_actual not in nodos_explorados:
            nodos_explorados.add(nodo_actual)

            for vecino, peso in red_vial.get(nodo_actual, []):
                nuevo_costo = costo_actual + peso
                if vecino not in costos_acumulados or nuevo_costo < costos_acumulados[vecino]:
                    costos_acumulados[vecino] = nuevo_costo
                    heapq.heappush(frontera, (nuevo_costo, vecino))

    return float('inf')

# Representación del grafo ponderado
mapa_rutas = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('D', 2), ('E', 5)],
    'C': [('A', 4), ('F', 2)],
    'D': [('B', 2)],
    'E': [('B', 5), ('F', 1)],
    'F': [('C', 2), ('E', 1)]
}

# Buscar el costo mínimo del nodo 'A' al nodo 'F'
costo_minimo = encontrar_camino_minimo(mapa_rutas, 'A', 'F')
print(f"El costo más bajo para ir de 'A' a 'F' es: {costo_minimo}")
