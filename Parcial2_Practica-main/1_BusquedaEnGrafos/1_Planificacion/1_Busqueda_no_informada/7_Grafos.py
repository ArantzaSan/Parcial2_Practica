from collections import deque

def esquema_general_busqueda(graph, start, goal, method='BFS'):
    # Inicializar la estructura de datos de la frontera según el método
    if method == 'BFS':
        estructura_frontera = deque([(start, [start])])
    elif method == 'DFS':
        estructura_frontera = [(start, [start])]
    else:
        raise ValueError("Método no reconocido. Debe ser 'BFS' o 'DFS'.")

    # Conjunto para rastrear nodos ya explorados
    nodos_explorados = set()

    while estructura_frontera:
        # Obtener el siguiente nodo y su camino según el método
        if method == 'BFS':
            nodo_actual, camino_actual = estructura_frontera.popleft()
        elif method == 'DFS':
            nodo_actual, camino_actual = estructura_frontera.pop()

        # Si el nodo actual es el objetivo, retornar el camino
        if nodo_actual == goal:
            return camino_actual

        # Marcar el nodo actual como explorado
        if nodo_actual not in nodos_explorados:
            nodos_explorados.add(nodo_actual)

            # Añadir vecinos no explorados a la frontera
            for vecino in graph[nodo_actual]:
                if vecino not in nodos_explorados:
                    nuevo_camino = camino_actual + [vecino]
                    estructura_frontera.append((vecino, nuevo_camino))

    # Si la frontera se vacía sin encontrar el objetivo
    return None

# Ejemplo de la estructura del grafo
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# Ejecutar la búsqueda usando BFS
ruta_bfs = general_uninformed_search(graph, 'A', 'F', strategy='BFS')
print(f"Ruta encontrada con BFS: {ruta_bfs}")

# Ejecutar la búsqueda usando DFS
ruta_dfs = general_uninformed_search(graph, 'A', 'F', strategy='DFS')
print(f"Ruta encontrada con DFS: {ruta_dfs}")
