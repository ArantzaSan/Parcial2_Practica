from collections import deque

def general_uninformed_search(graph, start, goal, strategy='BFS'):
    # Seleccionamos la estrategia de búsqueda
    if strategy == 'BFS':
        frontier = deque([(start, [start])])  # Usamos una cola para BFS
    elif strategy == 'DFS':
        frontier = [(start, [start])]  # Usamos una pila para DFS
    else:
        raise ValueError("Estrategia no soportada. Usa 'BFS' o 'DFS'.")

    # Conjunto para mantener el registro de los nodos visitados
    visited = set()

    while frontier:
        # Sacamos el nodo de la frontera según la estrategia
        if strategy == 'BFS':
            node, path = frontier.popleft()
        elif strategy == 'DFS':
            node, path = frontier.pop()

        # Si alcanzamos el nodo objetivo, devolvemos el camino
        if node == goal:
            return path

        # Si el nodo no ha sido visitado
        if node not in visited:
            visited.add(node)

            # Agregamos todos los vecinos no visitados a la frontera
            for neighbor in graph[node]:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    frontier.append((neighbor, new_path))

    # Si no encontramos el nodo objetivo, devolvemos None
    return None

# Ejemplo de grafo representado como un diccionario de listas
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# Ejecutamos la búsqueda no informada desde el nodo 'A' hasta el nodo 'F' usando BFS
bfs_path = general_uninformed_search(graph, 'A', 'F', strategy='BFS')
print(f"Camino desde 'A' hasta 'F' usando BFS: {bfs_path}")

# Ejecutamos la búsqueda no informada desde el nodo 'A' hasta el nodo 'F' usando DFS
dfs_path = general_uninformed_search(graph, 'A', 'F', strategy='DFS')
print(f"Camino desde 'A' hasta 'F' usando DFS: {dfs_path}")
