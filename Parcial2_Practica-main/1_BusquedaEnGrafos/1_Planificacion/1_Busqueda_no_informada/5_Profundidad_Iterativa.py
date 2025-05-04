def buscar_profundidad_iterativa(grafo_conexo, inicio_nodo, nodo_final):
    """
    Implementa la búsqueda en profundidad iterativa (IDDFS) para encontrar un
    camino desde un nodo inicial hasta un nodo final en un grafo.

    Args:
        grafo_conexo (dict): Un diccionario que representa el grafo.
                                Las claves son los nodos y los valores son listas de vecinos.
        inicio_nodo: El nodo desde donde comienza la búsqueda.
        nodo_final: El nodo objetivo que se desea encontrar.

    Returns:
        list or None: Una lista que representa el camino encontrado desde el
                     nodo de inicio hasta el nodo final. Retorna None si no
                     se encuentra ningún camino.
    """

    def _dfs_con_limite(grafo, nodo_actual, objetivo, profundidad_restante, historial_ruta):
        """Función auxiliar para realizar la búsqueda en profundidad limitada."""
        if nodo_actual == objetivo:
            return historial_ruta

        if profundidad_restante <= 0:
            return None

        for vecino in grafo.get(nodo_actual, []):
            if vecino not in historial_ruta:
                resultado = _dfs_con_limite(grafo, vecino, objetivo, profundidad_restante - 1, historial_ruta + [vecino])
                if resultado:
                    return resultado
        return None

    profundidad = 0
    while True:
        camino = _dfs_con_limite(grafo_conexo, inicio_nodo, nodo_final, profundidad, [inicio_nodo])
        if camino:
            return camino
        profundidad += 1

# Ejemplo de la estructura del grafo
mi_grafo_ejemplo = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# Ejecutar la búsqueda en profundidad iterativa desde 'A' hasta 'F'
ruta_hallada = buscar_profundidad_iterativa(mi_grafo_ejemplo, 'A', 'F')
print(f"Ruta encontrada usando IDDFS (A a F): {ruta_hallada}")
