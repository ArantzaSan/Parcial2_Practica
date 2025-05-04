def buscar_en_profundidad(estructura_grafo, inicio_nodo, nodo_objetivo):
    """
    Realiza una búsqueda en profundidad (DFS) para encontrar un camino desde un
    nodo de inicio hasta un nodo objetivo en un grafo.

    Args:
        estructura_grafo (dict): Un diccionario que representa el grafo.
                                   Las claves son los nodos y los valores son listas de vecinos.
        inicio_nodo: El nodo desde donde comienza la búsqueda.
        nodo_objetivo: El nodo que se desea encontrar.

    Returns:
        list or None: Una lista que representa el camino encontrado desde el
                     nodo de inicio hasta el nodo objetivo. Retorna None si no
                     se encuentra ningún camino.
    """
    pila_exploracion = [(inicio_nodo, [inicio_nodo])]
    nodos_visitados = set()

    while pila_exploracion:
        nodo_actual, ruta_actual = pila_exploracion.pop()

        if nodo_actual == nodo_objetivo:
            return ruta_actual

        if nodo_actual not in nodos_visitados:
            nodos_visitados.add(nodo_actual)
            conexiones = estructura_grafo.get(nodo_actual, [])
            for vecino in conexiones:
                if vecino not in nodos_visitados:
                    pila_exploracion.append((vecino, ruta_actual + [vecino]))

    return None

# Definición del grafo como un diccionario
mi_red = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# Buscar un camino desde 'A' hasta 'F' usando DFS
camino_encontrado = buscar_en_profundidad(mi_red, 'A', 'F')
print(f"El camino encontrado desde 'A' hasta 'F' es: {camino_encontrado}")
