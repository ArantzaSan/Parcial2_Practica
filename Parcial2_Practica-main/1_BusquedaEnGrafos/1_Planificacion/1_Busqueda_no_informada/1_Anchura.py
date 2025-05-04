from collections import deque

def explorar_grafo_anchura(estructura_grafo, nodo_inicial):
    """
    Realiza una búsqueda en anchura (BFS) en un grafo.

    Args:
        estructura_grafo (dict): Un diccionario que representa el grafo.
                                   Las claves son los nodos y los valores son listas de vecinos.
        nodo_inicial: El nodo desde donde comienza la búsqueda.

    Returns:
        set: Un conjunto que contiene todos los nodos visitados durante la búsqueda.
    """
    cola_exploracion = deque([nodo_inicial])
    nodos_visitados = set()

    while cola_exploracion:
        nodo_actual = cola_exploracion.popleft()

        if nodo_actual not in nodos_visitados:
            print(f"Analizando el nodo: {nodo_actual}")
            nodos_visitados.add(nodo_actual)
            vecinos = estructura_grafo.get(nodo_actual, []) # Obtener vecinos de forma segura
            for vecino in vecinos:
                if vecino not in nodos_visitados:
                    cola_exploracion.append(vecino)

    return nodos_visitados

# Definición del grafo como un diccionario
mi_grafo = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# Ejecutar la búsqueda en anchura desde el nodo 'A'
nodos_alcanzados = explorar_grafo_anchura(mi_grafo, 'A')
print(f"\nNodos visitados en total: {nodos_alcanzados}")
